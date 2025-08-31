# consistency_healthcare

## System Monitoring

To monitor system resources (memory and CPU usage) during model inference, use:

```bash
watch -n 2 "echo 'System Monitor - $(date)'; echo ''; echo 'Memory Usage:'; free -h | awk 'NR==1{print \"Type|Total|Used|Free|Shared|Buff/Cache|Available\"} NR>1{print \$1\"|\"\$2\"|\"\$3\"|\"\$4\"|\"\$5\"|\"\$6\"|\"\$7}'; echo ''; echo 'CPU Usage:'; top -bn1 | grep 'Cpu(s)' | awk '{print \"CPU Usage: \" 100-\$8 \"%\"}'"
```

This command refreshes every 2 seconds and displays:
- Timestamp
- Memory usage in pipe-separated format
- CPU usage percentage

To stop monitoring, press `Ctrl+C` or run `pkill -f "watch -n 2"`.