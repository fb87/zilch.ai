//! Print the repository map of a directory: `cargo run -p zilch-core --example repomap -- [dir] [path]`.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = std::path::PathBuf::from(args.first().map(String::as_str).unwrap_or("."));
    let t = std::time::Instant::now();
    let map = zilch_core::repomap::RepoMap::build(&root);
    let built = t.elapsed();
    match args.get(1) {
        Some(p) => println!("{}", map.detail(p, 12_000)),
        None => println!("{}", map.overview(8_000)),
    }
    eprintln!("(built in {built:?})");
}
