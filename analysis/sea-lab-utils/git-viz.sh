#!/bin/bash
# first remove branches that have been deleted on remote
git remote prune origin

# show simplified tree of remote branches only
git log --graph --decorate --oneline --simplify-by-decoration --remotes --decorate-refs=refs/remotes/
