//! Print the repository map of a directory: `cargo run -p zilch-core --example repomap -- [dir] [path]`.
//! With `--cached`, time loading the disk cache and refreshing it (what a session start does).
fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let cached = args.iter().any(|a| a == "--cached");
    args.retain(|a| a != "--cached");
    let root = std::path::PathBuf::from(args.first().map(String::as_str).unwrap_or("."));
    let t = std::time::Instant::now();
    let map = if cached {
        let (mut map, hit) = zilch_core::repomap::RepoMap::load_or_build(&root);
        eprintln!("(cache {}: {:?})", if hit { "hit" } else { "miss, built" }, t.elapsed());
        let t = std::time::Instant::now();
        map.refresh(&root, "");
        eprintln!("(refreshed in {:?})", t.elapsed());
        map
    } else {
        zilch_core::repomap::RepoMap::build(&root)
    };
    eprintln!("(total {:?})", t.elapsed());
    match args.get(1) {
        Some(p) => println!("{}", map.detail(p, 12_000)),
        None => println!("{}", map.overview(8_000)),
    }
}
