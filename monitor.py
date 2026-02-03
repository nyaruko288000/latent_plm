"""
轻量级训练监控
- 记录 metrics 到 JSON
- 提供 HTTP 服务
- 手机友好的前端
"""

import json
import threading
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import subprocess
import os


@dataclass
class TrainingMetrics:
    """训练指标存储"""
    steps: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    loss_wm: List[float] = field(default_factory=list)
    loss_decoder: List[float] = field(default_factory=list)
    timestamps: List[str] = field(default_factory=list)
    
    # 当前状态
    current_step: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    samples_per_sec: float = 0.0
    eta_seconds: float = 0.0
    status: str = "initializing"


class MetricsLogger:
    """指标记录器"""
    
    def __init__(self, save_dir: str = "logs"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.metrics = TrainingMetrics()
        self.metrics_file = self.save_dir / "metrics.json"
        
        self._last_step_time = time.time()
        self._last_step = 0
        
    def log(
        self,
        step: int,
        train_loss: Optional[float] = None,
        train_acc: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_acc: Optional[float] = None,
        lr: Optional[float] = None,
        loss_wm: Optional[float] = None,
        loss_decoder: Optional[float] = None,
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
    ):
        """记录一条指标"""
        now = datetime.now().strftime("%H:%M:%S")
        
        if train_loss is not None:
            self.metrics.steps.append(step)
            self.metrics.train_loss.append(train_loss)
            self.metrics.timestamps.append(now)
            
            if train_acc is not None:
                self.metrics.train_acc.append(train_acc)
            if lr is not None:
                self.metrics.lr.append(lr)
            if loss_wm is not None:
                self.metrics.loss_wm.append(loss_wm)
            if loss_decoder is not None:
                self.metrics.loss_decoder.append(loss_decoder)
        
        if val_loss is not None:
            self.metrics.val_loss.append(val_loss)
        if val_acc is not None:
            self.metrics.val_acc.append(val_acc)
        
        # 更新状态
        self.metrics.current_step = step
        if epoch is not None:
            self.metrics.current_epoch = epoch
        if total_epochs is not None:
            self.metrics.total_epochs = total_epochs
        
        # 计算速度
        elapsed = time.time() - self._last_step_time
        if elapsed > 0 and step > self._last_step:
            self.metrics.samples_per_sec = (step - self._last_step) / elapsed
        
        self.metrics.status = "training"
        
        # 保存到文件
        self._save()
    
    def set_status(self, status: str):
        self.metrics.status = status
        self._save()
    
    def _save(self):
        with open(self.metrics_file, "w") as f:
            json.dump(asdict(self.metrics), f)


# HTML 前端（手机友好）
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>🚀 Training Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 16px;
        }
        .header {
            text-align: center;
            padding: 20px 0;
        }
        .header h1 {
            font-size: 24px;
            margin-bottom: 8px;
        }
        .status {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .status.training { background: #4ecca3; color: #1a1a2e; }
        .status.initializing { background: #ffd93d; color: #1a1a2e; }
        .status.completed { background: #6c63ff; }
        .status.error { background: #ff6b6b; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin: 20px 0;
        }
        .metric-card {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 16px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }
        .metric-card.wide {
            grid-column: span 2;
        }
        .metric-label {
            font-size: 12px;
            color: #888;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #4ecca3;
        }
        .metric-value.loss { color: #ff6b6b; }
        .metric-value.acc { color: #4ecca3; }
        .metric-value.lr { color: #ffd93d; }
        
        .chart-container {
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 16px;
            margin: 16px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .chart-title {
            font-size: 14px;
            color: #888;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        canvas {
            max-height: 200px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }
        
        .refresh-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4ecca3;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
    </style>
</head>
<body>
    <div class="refresh-indicator"></div>
    
    <div class="header">
        <h1>🚀 Training Monitor</h1>
        <span class="status" id="status">Loading...</span>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Step</div>
            <div class="metric-value" id="step">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Epoch</div>
            <div class="metric-value" id="epoch">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Train Loss</div>
            <div class="metric-value loss" id="train_loss">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Train Acc</div>
            <div class="metric-value acc" id="train_acc">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Val Loss</div>
            <div class="metric-value loss" id="val_loss">-</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Val Acc</div>
            <div class="metric-value acc" id="val_acc">-</div>
        </div>
        <div class="metric-card wide">
            <div class="metric-label">Learning Rate</div>
            <div class="metric-value lr" id="lr">-</div>
        </div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📉 Loss Curves</div>
        <canvas id="lossChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">📈 Accuracy</div>
        <canvas id="accChart"></canvas>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">🔧 Planner vs Decoder Loss</div>
        <canvas id="componentsChart"></canvas>
    </div>
    
    <div class="footer">
        Last updated: <span id="lastUpdate">-</span><br>
        Speed: <span id="speed">-</span> steps/sec
    </div>

    <script>
        // Chart.js 配置
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';
        
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: true,
            animation: { duration: 0 },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12, padding: 8, font: { size: 11 } }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: { display: false },
                    ticks: { maxTicksLimit: 5, font: { size: 10 } }
                },
                y: {
                    display: true,
                    ticks: { maxTicksLimit: 5, font: { size: 10 } }
                }
            },
            elements: {
                point: { radius: 0 },
                line: { borderWidth: 2 }
            }
        };
        
        // 初始化图表
        const lossChart = new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Train', data: [], borderColor: '#ff6b6b', fill: false },
                    { label: 'Val', data: [], borderColor: '#ffd93d', fill: false }
                ]
            },
            options: chartOptions
        });
        
        const accChart = new Chart(document.getElementById('accChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'Train', data: [], borderColor: '#4ecca3', fill: false },
                    { label: 'Val', data: [], borderColor: '#6c63ff', fill: false }
                ]
            },
            options: chartOptions
        });
        
        const componentsChart = new Chart(document.getElementById('componentsChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: 'World Model', data: [], borderColor: '#ff9f43', fill: false },
                    { label: 'Decoder', data: [], borderColor: '#00d2d3', fill: false }
                ]
            },
            options: chartOptions
        });
        
        // 采样函数（避免太多数据点）
        function sampleData(arr, maxPoints = 100) {
            if (arr.length <= maxPoints) return arr;
            const step = Math.ceil(arr.length / maxPoints);
            return arr.filter((_, i) => i % step === 0);
        }
        
        // 更新数据
        async function updateData() {
            try {
                const response = await fetch('/metrics.json?' + Date.now());
                const data = await response.json();
                
                // 更新状态
                const statusEl = document.getElementById('status');
                statusEl.textContent = data.status;
                statusEl.className = 'status ' + data.status;
                
                // 更新指标
                document.getElementById('step').textContent = data.current_step.toLocaleString();
                document.getElementById('epoch').textContent = 
                    `${data.current_epoch + 1}/${data.total_epochs || '?'}`;
                
                const lastTrainLoss = data.train_loss[data.train_loss.length - 1];
                const lastTrainAcc = data.train_acc[data.train_acc.length - 1];
                const lastValLoss = data.val_loss[data.val_loss.length - 1];
                const lastValAcc = data.val_acc[data.val_acc.length - 1];
                const lastLr = data.lr[data.lr.length - 1];
                
                document.getElementById('train_loss').textContent = 
                    lastTrainLoss ? lastTrainLoss.toFixed(4) : '-';
                document.getElementById('train_acc').textContent = 
                    lastTrainAcc ? (lastTrainAcc * 100).toFixed(1) + '%' : '-';
                document.getElementById('val_loss').textContent = 
                    lastValLoss ? lastValLoss.toFixed(4) : '-';
                document.getElementById('val_acc').textContent = 
                    lastValAcc ? (lastValAcc * 100).toFixed(1) + '%' : '-';
                document.getElementById('lr').textContent = 
                    lastLr ? lastLr.toExponential(2) : '-';
                
                document.getElementById('speed').textContent = 
                    data.samples_per_sec ? data.samples_per_sec.toFixed(1) : '-';
                document.getElementById('lastUpdate').textContent = 
                    new Date().toLocaleTimeString();
                
                // 更新图表
                const sampledSteps = sampleData(data.steps);
                const indices = sampledSteps.map((_, i) => 
                    Math.floor(i * data.steps.length / sampledSteps.length));
                
                // Loss chart
                lossChart.data.labels = sampledSteps;
                lossChart.data.datasets[0].data = indices.map(i => data.train_loss[i]);
                lossChart.data.datasets[1].data = sampleData(data.val_loss);
                lossChart.update('none');
                
                // Accuracy chart
                accChart.data.labels = sampledSteps;
                accChart.data.datasets[0].data = indices.map(i => data.train_acc[i]);
                accChart.data.datasets[1].data = sampleData(data.val_acc);
                accChart.update('none');
                
                // Components chart
                if (data.loss_wm && data.loss_wm.length > 0) {
                    componentsChart.data.labels = sampledSteps;
                    componentsChart.data.datasets[0].data = indices.map(i => data.loss_wm[i]);
                    componentsChart.data.datasets[1].data = indices.map(i => data.loss_decoder[i]);
                    componentsChart.update('none');
                }
                
            } catch (e) {
                console.error('Failed to fetch metrics:', e);
            }
        }
        
        // 每 3 秒更新一次
        updateData();
        setInterval(updateData, 3000);
    </script>
</body>
</html>
"""


class MonitorServer:
    """监控 HTTP 服务器"""
    
    def __init__(self, log_dir: str = "logs", port: int = 8080):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.port = port
        self.server = None
        self.thread = None
        self.tunnel_url = None
        
        # 写入 HTML
        (self.log_dir / "index.html").write_text(DASHBOARD_HTML)
        
        # 初始化空 metrics
        if not (self.log_dir / "metrics.json").exists():
            with open(self.log_dir / "metrics.json", "w") as f:
                json.dump(asdict(TrainingMetrics()), f)
    
    def start(self):
        """启动服务器"""
        os.chdir(self.log_dir)
        
        handler = SimpleHTTPRequestHandler
        handler.log_message = lambda *args: None  # 禁用日志
        
        self.server = HTTPServer(("0.0.0.0", self.port), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        
        print(f"📊 Monitor server started at http://localhost:{self.port}")
    
    def start_tunnel(self) -> str:
        """启动 Cloudflare 隧道"""
        try:
            # 检查是否已安装 cloudflared
            result = subprocess.run(
                ["which", "cloudflared"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode != 0:
                print("Installing cloudflared...")
                subprocess.run([
                    "wget", "-q", 
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    "-O", "/tmp/cloudflared"
                ], check=True)
                subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=True)
                cloudflared_path = "/tmp/cloudflared"
            else:
                cloudflared_path = "cloudflared"
            
            # 启动隧道
            process = subprocess.Popen(
                [cloudflared_path, "tunnel", "--url", f"http://localhost:{self.port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待获取 URL
            import time
            for _ in range(30):
                time.sleep(1)
                try:
                    # 读取 stderr 获取 URL
                    line = process.stderr.readline()
                    if "trycloudflare.com" in line or ".cloudflare" in line:
                        # 提取 URL
                        import re
                        match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
                        if match:
                            self.tunnel_url = match.group(0)
                            print(f"\n{'='*50}")
                            print(f"📱 手机访问地址:")
                            print(f"   {self.tunnel_url}")
                            print(f"{'='*50}\n")
                            return self.tunnel_url
                except:
                    pass
            
            print("Warning: Could not get tunnel URL")
            return None
            
        except Exception as e:
            print(f"Warning: Failed to start tunnel: {e}")
            return None
    
    def stop(self):
        """停止服务器"""
        if self.server:
            self.server.shutdown()


def start_monitor(log_dir: str = "logs", port: int = 8080, use_tunnel: bool = True):
    """启动监控服务"""
    server = MonitorServer(log_dir, port)
    server.start()
    
    if use_tunnel:
        server.start_tunnel()
    
    return server