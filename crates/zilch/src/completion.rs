//! Tab completion for the REPL: slash commands, their arguments, and `@` mentions. A pure
//! function over what is live right now (pids, models), so it is tested without a terminal.

/// Every REPL command, for completion (keep in step with /help).
pub const COMMANDS: &[&str] = &[
    "/ack",
    "/about",
    "/cancel",
    "/clear",
    "/code",
    "/exit",
    "/help",
    "/kill",
    "/model",
    "/models",
    "/prompt",
    "/ps",
    "/quiet",
    "/quit",
    "/recap",
    "/redraw",
    "/sessions",
    "/spec",
    "/status",
    "/usage",
    "/verbose",
];

/// What completion can offer besides the fixed words.
#[derive(Debug, Clone, Default)]
pub struct Live {
    /// Live process ids.
    pub pids: Vec<u64>,
    /// Model ids the providers list (without the provider prefix).
    pub models: Vec<String>,
}

/// Candidates for the word ending at `pos`: (where the word starts, candidates). Commands and
/// complete arguments get a trailing space so the next word can follow at once.
pub fn complete(line: &str, pos: usize, live: &Live) -> (usize, Vec<String>) {
    let head = &line[..pos];
    let start = head.rfind(char::is_whitespace).map_or(0, |i| i + 1);
    let word = &head[start..];
    let earlier: Vec<&str> = head[..start].split_whitespace().collect();
    let pick = |options: Vec<String>, space: bool| -> Vec<String> {
        let mut v: Vec<String> =
            options.into_iter().filter(|o| o.starts_with(word)).map(|o| if space { format!("{o} ") } else { o }).collect();
        v.sort();
        v.dedup();
        v
    };
    let words = |w: &[&str]| w.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let pids = || live.pids.iter().map(u64::to_string).collect::<Vec<_>>();
    let candidates = match earlier.as_slice() {
        // The first word: a command, or an @ mention.
        [] if word.starts_with('/') => pick(words(COMMANDS), true),
        [] if word.starts_with('@') => {
            let mut targets = words(&["@main", "@developer", "@spec"]);
            targets.extend(live.pids.iter().map(|p| format!("@{p}")));
            pick(targets, true)
        }
        ["/model"] => pick(words(&["main", "developer"]), true),
        ["/model", "main" | "developer"] => pick(words(&["default"]).into_iter().chain(live.models.iter().cloned()).collect(), true),
        ["/prompt"] => pick(words(&["main", "answerer", "planner", "developer", "spec"]), false),
        ["/kill"] => pick(pids(), false),
        _ => vec![],
    };
    (start, candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live() -> Live {
        Live { pids: vec![1, 3, 12], models: vec!["beam-free".into(), "glm-5.3".into(), "glm-5.3-flash".into()] }
    }

    fn c(line: &str) -> Vec<String> {
        complete(line, line.len(), &live()).1
    }

    #[test]
    fn commands_and_their_arguments_complete() {
        assert_eq!(c("/mo"), ["/model ", "/models "]);
        assert_eq!(c("/s"), ["/sessions ", "/spec ", "/status "]);
        assert!(c("/").len() == COMMANDS.len());
        assert_eq!(c("/model "), ["developer ", "main "]);
        assert_eq!(c("/model d"), ["developer "]);
        assert_eq!(c("/model main glm"), ["glm-5.3 ", "glm-5.3-flash "]);
        assert_eq!(c("/model main de"), ["default "]);
        assert_eq!(c("/kill 1"), ["1", "12"]);
        assert_eq!(c("/prompt "), ["answerer", "developer", "main", "planner", "spec"]);
        assert!(c("/model main glm-5.3 x").is_empty(), "nothing after the last argument");
    }

    #[test]
    fn mentions_complete_groups_and_live_pids() {
        assert_eq!(c("@"), ["@1 ", "@12 ", "@3 ", "@developer ", "@main ", "@spec "]);
        assert_eq!(c("@m"), ["@main "]);
        assert_eq!(c("@1"), ["@1 ", "@12 "]);
        assert!(c("@main fix th").is_empty(), "the note itself is free text");
    }

    #[test]
    fn plain_orders_are_not_completed_and_the_word_start_is_reported() {
        assert!(c("add a flag").is_empty());
        assert_eq!(complete("/model main gl", 14, &live()).0, 12);
    }
}
