use anyhow::{anyhow, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use serde_json::Value as JsonValue;
use std::env;

/// Call a Python shim module for the specified engine.
/// engine: "deepmind" or "ligo"
pub fn call_shim(engine: &str, passes: u32, device: &str, seq_len: u32, notes: Option<&str>, full: bool, mode: &str) -> Result<JsonValue> {
    Python::with_gil(|py| {
        // Ensure venv site-packages and our project `py/` are on sys.path
        // scripts/dev.sh sets venv and PYTHONPATH; no sys.path mutation needed here
        // (no local sys.path mutation)
        let module_name = match engine {
            "deepmind" => "deepmind_shim",
            "ligo" => "ligo_shim",
            other => return Err(anyhow!("Unknown engine: {}", other)),
        };

        let shim = PyModule::import_bound(py, module_name)?;

        // Prefer forward_once; fall back to hello for the earliest MVP
        let callable = match shim.getattr("forward_once") {
            Ok(func) => func,
            Err(_) => shim.getattr("hello")?,
        };

        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("passes", passes)?;
        kwargs.set_item("device", device)?;
        kwargs.set_item("seq_len", seq_len)?;
        if let Some(n) = notes { kwargs.set_item("notes", n)?; }
        kwargs.set_item("full", full)?;
        kwargs.set_item("mode", mode)?;

        let py_result = callable.call((), Some(&kwargs))?;

        // Convert Python object to JSON via json.dumps to avoid extra dependencies
        let json_mod = PyModule::import_bound(py, "json")?;
        let json_str: String = json_mod.call_method1("dumps", (py_result,))?.extract()?;
        let value: JsonValue = serde_json::from_str(&json_str)?;
        Ok(value)
    })
}

