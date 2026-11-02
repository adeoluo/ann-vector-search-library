# Records docs/demo/benchmark.gif from a real run: `vhs docs/demo/benchmark.tape` from the repo root.
Output docs/demo/benchmark.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 460
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m ann_vector_search_library benchmark` Enter
Sleep 4s
