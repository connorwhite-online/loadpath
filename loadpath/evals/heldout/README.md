# Held-out parts: the cold benchmark

Common parts nobody has tuned loadpath on: a corner bracket, a hinge leaf, a
drawer pull, a VESA plate, a caster fork, a pipe clamp, a crank lever, a
tablet stand, a C-clamp, an action-camera mount. Each study was written once,
cold, from SKILL.md alone, the way an agent meeting loadpath for the first
time would write it, and run without looking at loadpath's code.

The question the set answers: **does a part loadpath has never seen come out
right the first time?** The eval cases (`loadpath/evals/cases`) cannot answer
it, because the library has been iterated against them.

## Rules

1. **Never tune against this set.** No case-specific constants, no threshold
   picked so a case here passes. A fix is made for a cause that follows from
   physics or from the problem statement, checked on other studies, and only
   then measured here.
2. **Never edit a case to make it pass.** If a case turns out to be a wrong
   problem statement (a goal no material in that space can meet, an envelope
   that cannot hold a path), it stays: an agent will write the same mistake,
   and the test is whether loadpath says so quickly and usefully. Such cases
   carry a `note` and `pass.diagnosis`. A corrected twin may be added beside
   it under a new id.
3. **Add, do not replace.** New cases are written cold and their first run
   is recorded below before anything is changed for them.
4. **First try means first try.** `firstTry` is: `loadpath check` is clean
   (no warnings) and `loadpath run` delivers a part that passes, from the
   study as written.

## Running

```
python -m loadpath.evals.heldout list
python -m loadpath.evals.heldout run --out runs/NAME --jobs 2 [ids...]
python -m loadpath.evals.heldout table runs/NAME
```

## Run history

RUN_HISTORY
