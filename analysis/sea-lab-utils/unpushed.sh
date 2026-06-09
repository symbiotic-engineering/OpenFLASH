#!/bin/bash
# Bash script to show git commits that haven't been pushed.
# Checks all of the branches of all of the git repos that are under the current directory.

# Find all .git directories
find . -type d -iname '.git' -exec sh -c '
    # Change to the parent directory of the .git folder
    cd "${0}/../" || exit

    # Check if there are any commits in branches not on remotes
    if git log --branches --not --remotes --no-walk --decorate --oneline | grep -q ""; then
        echo ""  # Print an empty line
        pwd      # Print the current directory
        # Print the log of commits
        git log --branches --not --remotes --no-walk --decorate --oneline
    fi
' "{}" \;

# code modified from https://stackoverflow.com/a/33391814 and https://stackoverflow.com/a/48180899
