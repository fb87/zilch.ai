//! Standalone sandbox helper (the `zilch` CLI embeds the same entry point as `zilch __sandbox`).
//! Usage: zilch-sandbox <helper-spec-json> -- <command> [args…]

fn main() {
    #[cfg(target_os = "linux")]
    zilch_drivers::tools::sandbox_linux::helper_main(std::env::args().skip(1));
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("zilch-sandbox: only supported on Linux");
        std::process::exit(126);
    }
}
