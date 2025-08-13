use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::Value as JsonValue;
use chrono::Utc;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use sysinfo::System;
use csv::Writer;

#[derive(Parser, Debug)]
#[command(name = "af3-bench", version, about = "AF3 Bench CLI", author = "")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run inference timing
    Run {
        /// Engine to run: deepmind, ligo, or both
        #[arg(long, value_enum, default_value_t = EngineArg::Both)]
        engine: EngineArg,

        /// Number of passes
        #[arg(long)]
        passes: Option<u32>,

        /// Alias for --passes 1
        #[arg(long, default_value_t = false)]
        dry_run: bool,

        /// Device to target (cpu for now)
        #[arg(long, default_value = "cpu")]
        device: String,

        /// Sequence length (small for quick CPU tests)
        #[arg(long, default_value_t = 32)]
        seq_len: u32,

        /// Notes to attach to the run
        #[arg(long)]
        notes: Option<String>,

        /// Output directory (default: results/<timestamp>)
        #[arg(long)]
        out: Option<PathBuf>,

        /// Full end-to-end path (includes diffusion). On macOS MPS we keep trunk-only.
        #[arg(long, default_value_t = false)]
        full: bool,

        /// Mode: toy|trunk|full (toy = orchestration sanity)
        #[arg(long, default_value = "trunk")]
        mode: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, ValueEnum)]
enum EngineArg {
    Deepmind,
    Ligo,
    Both,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run { engine, passes, dry_run, device, seq_len, notes, out, full, mode } => {
            let passes = if dry_run { 1 } else { passes.unwrap_or(1) };
            let engines: Vec<&str> = match engine {
                EngineArg::Deepmind => vec!["deepmind"],
                EngineArg::Ligo => vec!["ligo"],
                EngineArg::Both => vec!["deepmind", "ligo"],
            };

            // Prepare output directory
            let ts = Utc::now().format("%Y%m%d-%H%M%S").to_string();
            let out_dir = out.unwrap_or_else(|| PathBuf::from(format!("results/{}", ts)));
            fs::create_dir_all(&out_dir).with_context(|| format!("creating {:?}", out_dir))?;

            // Collect host + commits metadata
            let (dm_commit, ligo_commit) = (git_rev("third_party/alphafold3-deepmind"), git_rev("third_party/alphafold3-ligo"));
            let sys = System::new_all();
            let cpu_brand = sys
                .cpus()
                .get(0)
                .map(|c| c.brand().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            // CSV summary writer
            let mut csvw = Writer::from_path(out_dir.join("summary.csv"))?;
            csvw.write_record(["engine","pass_index","start_ts","elapsed_ms","device","seq_len","commit","notes","cpu_brand"]) ?;

            for eng in engines {
                let result: JsonValue = af3_core::call_shim(eng, passes, &device, seq_len, notes.as_deref(), full, &mode)?;

                // Open JSONL per engine
                let jsonl_path = out_dir.join(format!("{}.jsonl", eng));
                let mut jsonl = BufWriter::new(File::create(&jsonl_path)?);

                // Expect result is a list of dicts
                let arr = result.as_array().cloned().unwrap_or_default();
                let commit = match eng {
                    "deepmind" => dm_commit.as_deref().unwrap_or("unknown"),
                    "ligo" => ligo_commit.as_deref().unwrap_or("unknown"),
                    _ => "unknown",
                };
                let mut elapsed: Vec<f64> = Vec::with_capacity(arr.len());
                for item in arr {
                    // augment item with commit and cpu_brand for JSONL and write CSV row
                    let mut obj = item.as_object().cloned().unwrap_or_default();
                    obj.insert("commit".into(), JsonValue::String(commit.to_string()));
                    obj.insert("cpu_brand".into(), JsonValue::String(cpu_brand.clone()));
                    let line = serde_json::to_string(&JsonValue::Object(obj.clone()))?;
                    writeln!(jsonl, "{}", line)?;

                    let engine = obj.get("engine").and_then(|v| v.as_str()).unwrap_or(eng);
                    let pass_index = obj.get("pass_index").and_then(|v| v.as_i64()).unwrap_or(0).to_string();
                    let start_ts = obj.get("start_ts").and_then(|v| v.as_str()).unwrap_or("");
                    let elapsed_ms = obj.get("elapsed_ms").and_then(|v| v.as_f64()).unwrap_or(0.0).to_string();
                    if let Some(val) = obj.get("elapsed_ms").and_then(|v| v.as_f64()) { elapsed.push(val); }
                    let device = obj.get("device").and_then(|v| v.as_str()).unwrap_or("");
                    let seq_len_s = obj.get("seq_len").and_then(|v| v.as_u64()).unwrap_or(0).to_string();
                    let notes_s = obj.get("notes").and_then(|v| v.as_str()).unwrap_or("");
                    csvw.write_record([engine, &pass_index, start_ts, &elapsed_ms, device, &seq_len_s, commit, notes_s, &cpu_brand])?;
                }
                csvw.flush()?;

                // Print console summary
                if !elapsed.is_empty() {
                    elapsed.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n = elapsed.len();
                    let min = elapsed[0];
                    let median = percentile(&elapsed, 50.0);
                    let p95 = percentile(&elapsed, 95.0);
                    println!(
                        "engine={} passes={} min={:.3}ms median={:.3}ms p95={:.3}ms -> {:?}",
                        eng,
                        n,
                        min,
                        median,
                        p95,
                        jsonl_path
                    );
                } else {
                    println!("engine={} no results -> {:?}", eng, jsonl_path);
                }
            }
            println!("summary -> {:?}", out_dir.join("summary.csv"));
        }
    }
    Ok(())
}

fn git_rev(dir: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["-C", dir, "rev-parse", "HEAD"]) 
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let p = p.clamp(0.0, 100.0) / 100.0;
    let idx = p * (sorted_values.len() as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { return sorted_values[lo]; }
    let frac = idx - lo as f64;
    sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
}

