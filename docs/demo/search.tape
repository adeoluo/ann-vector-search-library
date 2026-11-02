# Records docs/demo/search.gif from a real run: `vhs docs/demo/search.tape` from the repo root.
Output docs/demo/search.gif
Set Shell "bash"
Set FontSize 15
Set Width 1300
Set Height 380
Set Padding 18
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms

Hide
Type `export PS1='$ ' && uv sync -q && source .venv/bin/activate && clear` Enter
Sleep 10s
Show
Type `python -m ann_vector_search_library search` Enter
Sleep 3s
