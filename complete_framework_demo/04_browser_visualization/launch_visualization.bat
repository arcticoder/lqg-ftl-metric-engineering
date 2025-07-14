@echo off
echo Launching FTL Ship Hull Geometry Visualization...
echo.
echo Opening in Chrome with WebGL support...
start chrome.exe --allow-file-access-from-files --enable-webgl --disable-web-security "file:///C:/Users/echo_/Code/asciimath/lqg-ftl-metric-engineering/complete_framework_demo/04_browser_visualization/ftl_hull_visualization.html"
echo.
echo If the visualization doesn't load:
echo 1. Ensure Chrome supports WebGL
echo 2. Check browser console for errors
echo 3. Try running from a local web server
pause
        