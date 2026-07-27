# Copy-under-write demo — renders .github/copy-under-write.gif
#
#   brew install vhs
#   demo/record.sh
#
# record.sh stages everything this tape needs at /tmp/valise-demo (a
# fixed path, so the tape stays directly runnable) and then calls vhs.
#
# The recording is real: the writer commits continuously for the whole
# take, the copies are plain `cp` with no coordination, and the verifier
# opens every one of them. No output here is staged.

Output .github/copy-under-write.gif

Set Shell zsh
Set FontSize 19
Set Width 1180
Set Height 640
Set Padding 22
Set Framerate 12
Set TypingSpeed 28ms

Set Theme { "background": "#0D0D0F", "foreground": "#F5F1EA", "cursor": "#C9884A", "selection": "#2A2A30", "black": "#0D0D0F", "red": "#D97757", "green": "#8FA86B", "yellow": "#C9884A", "blue": "#7E9CBF", "magenta": "#A98BB5", "cyan": "#7FAFA8", "white": "#F5F1EA", "brightBlack": "#8A857E", "brightRed": "#E08A6C", "brightGreen": "#A3BC80", "brightYellow": "#DBA267", "brightBlue": "#93AECE", "brightMagenta": "#BCA0C6", "brightCyan": "#93C0B9", "brightWhite": "#FFFFFF" }

Hide
Type "cd /tmp/valise-demo" Enter
Type "export PATH=/tmp/valise-demo/.venv/bin:/tmp/valise-demo/bin:$PATH" Enter
Type "setopt interactive_comments" Enter
Type "export PS1='%F{#C9884A}$%f '" Enter
Type "clear" Enter
Sleep 1s
Show

Type "# A writer is committing into corpus.vls, right now, without pause."
Enter
Sleep 1500ms

Type "python3 writer.py --out corpus.vls --stop-file .stop --delay 0.1 > writer.log &"
Enter
Sleep 4s

Type "tail -n 1 writer.log"
Enter
Sleep 2s

Type "# Copy it. No lock, no quiesce, no coordination — just cp."
Enter
Sleep 1500ms

Type "./copy_storm.sh corpus.vls copies 50"
Enter
Sleep 5s

Type "# The writer never stopped:"
Enter
Sleep 1000ms

Type "tail -n 1 writer.log"
Enter
Sleep 2s

Type "touch .stop"
Enter
Sleep 1500ms

Type "# So what did those 50 mid-write copies actually contain?"
Enter
Sleep 1500ms

Type "python3 verify_copies.py --glob 'copies/*.vls' --quiet"
Enter
Sleep 6s

Type "# Every one is a real capsule. Query one:"
Enter
Sleep 1500ms

Type "valise search copies/copy-07.vls docs 'vector quantization' --top-k 3"
Enter
Sleep 5s

Sleep 3s
