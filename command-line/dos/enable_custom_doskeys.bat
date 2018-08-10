@echo off

title Enabling custom doskeys...

doskey ll=ls
echo ll now alias for ls

doskey cl=cls
echo cl now an alias for cls

doskey superclear =cls cls cls cls cls cls
echo superclear now an alias for multiple cls

doskey gs=git status
echo gs now an alias for git status

doskey gd=git diff
echo gd now an alias for git diff

doskey ga=git add
echo ga now an alias for git add

doskey gp=git push
echo gp now an alias for git push

doskey gc=git commit
echo gc now an alias for git commit

pause
