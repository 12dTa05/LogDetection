using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LogAnomalyDetection
{
    /// <summary>
    /// Main form with Log Uploader Client and Server Monitor buttons.
    /// Matches original: "Log Uploader and Server Monitor" window
    /// </summary>
    public partial class MainForm : Form
    {
        private Button btnNewLogUploader;
        private Button btnOpenServerMonitor;
        private ComboBox cmbModelType;
        private Label lblModel;

        public MainForm()
        {
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.Text = "Log Uploader and Server Monitor";
            this.Size = new Size(420, 150);
            this.FormBorderStyle = FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.StartPosition = FormStartPosition.CenterScreen;

            // New Log Uploader Client button
            btnNewLogUploader = new Button
            {
                Text = "New Log Uploader Client",
                Location = new Point(20, 20),
                Size = new Size(170, 35),
                Font = new Font("Segoe UI", 9F)
            };
            btnNewLogUploader.Click += BtnNewLogUploader_Click;

            // Open Server Monitor button
            btnOpenServerMonitor = new Button
            {
                Text = "Open Server Monitor",
                Location = new Point(210, 20),
                Size = new Size(170, 35),
                Font = new Font("Segoe UI", 9F)
            };
            btnOpenServerMonitor.Click += BtnOpenServerMonitor_Click;

            // Model type dropdown
            lblModel = new Label
            {
                Text = "Model:",
                Location = new Point(20, 70),
                Size = new Size(50, 25),
                Font = new Font("Segoe UI", 9F)
            };

            cmbModelType = new ComboBox
            {
                Location = new Point(75, 68),
                Size = new Size(305, 25),
                DropDownStyle = ComboBoxStyle.DropDownList,
                Font = new Font("Segoe UI", 9F)
            };
            cmbModelType.Items.AddRange(new object[] { "Transformer", "CNN", "LSTM" });
            cmbModelType.SelectedIndex = 0;

            this.Controls.Add(btnNewLogUploader);
            this.Controls.Add(btnOpenServerMonitor);
            this.Controls.Add(lblModel);
            this.Controls.Add(cmbModelType);
        }

        private void BtnNewLogUploader_Click(object sender, EventArgs e)
        {
            var clientId = new Random().Next(1, 100);
            var form = new LogUploaderForm(cmbModelType.SelectedItem.ToString(), clientId);
            form.Show();
        }

        private void BtnOpenServerMonitor_Click(object sender, EventArgs e)
        {
            var form = new ServerMonitorForm();
            form.Show();
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            Application.Exit();
            base.OnFormClosing(e);
        }
    }

    /// <summary>
    /// Log Uploader Client form.
    /// Matches original: shows log processing in real-time with progress.
    /// </summary>
    public class LogUploaderForm : Form
    {
        private TextBox txtOutput;
        private Button btnUpload;
        private Button btnStopSending;
        private Button btnCheckConnection;
        private Label lblProcessed;
        private Label lblSendRate;
        private string modelType;
        private int clientId;
        private CancellationTokenSource cts;
        private int processedCount = 0;
        private static readonly HttpClient client = new HttpClient();
        private string serverUrl = "http://localhost:8000";

        public LogUploaderForm(string model, int id)
        {
            modelType = model;
            clientId = id;
            InitializeComponent();
        }

        private void InitializeComponent()
        {
            this.Text = $"Log Uploader - Client {clientId}";
            this.Size = new Size(700, 500);
            this.StartPosition = FormStartPosition.CenterScreen;

            // Output textbox
            txtOutput = new TextBox
            {
                Multiline = true,
                ScrollBars = ScrollBars.Vertical,
                Location = new Point(10, 10),
                Size = new Size(665, 350),
                ReadOnly = true,
                BackColor = Color.FromArgb(30, 30, 30),
                ForeColor = Color.LightGreen,
                Font = new Font("Consolas", 9F)
            };

            // Processed label
            lblProcessed = new Label
            {
                Text = "0 lines processed",
                Location = new Point(10, 370),
                Size = new Size(200, 25),
                Font = new Font("Segoe UI", 9F)
            };

            // Send rate label
            lblSendRate = new Label
            {
                Text = "Send Rate: 100 lines/second",
                Location = new Point(10, 440),
                Size = new Size(200, 25),
                Font = new Font("Segoe UI", 9F)
            };

            // Upload button
            btnUpload = new Button
            {
                Text = "Upload Log File",
                Location = new Point(220, 395),
                Size = new Size(130, 30),
                Font = new Font("Segoe UI", 9F)
            };
            btnUpload.Click += BtnUpload_Click;

            // Stop button
            btnStopSending = new Button
            {
                Text = "Stop Sending",
                Location = new Point(360, 395),
                Size = new Size(130, 30),
                Font = new Font("Segoe UI", 9F)
            };
            btnStopSending.Click += BtnStopSending_Click;

            // Check connection button
            btnCheckConnection = new Button
            {
                Text = "Check Connection",
                Location = new Point(500, 395),
                Size = new Size(130, 30),
                BackColor = Color.Green,
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 9F)
            };
            btnCheckConnection.Click += BtnCheckConnection_Click;

            this.Controls.Add(txtOutput);
            this.Controls.Add(lblProcessed);
            this.Controls.Add(lblSendRate);
            this.Controls.Add(btnUpload);
            this.Controls.Add(btnStopSending);
            this.Controls.Add(btnCheckConnection);
        }

        private async void BtnUpload_Click(object sender, EventArgs e)
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Filter = "Log files (*.log)|*.log|All files (*.*)|*.*";
                ofd.Title = "Select Log File";

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    cts = new CancellationTokenSource();
                    await ProcessLogFile(ofd.FileName, cts.Token);
                }
            }
        }

        private void BtnStopSending_Click(object sender, EventArgs e)
        {
            cts?.Cancel();
            AppendOutput("Stopped sending.");
        }

        private async void BtnCheckConnection_Click(object sender, EventArgs e)
        {
            try
            {
                var response = await client.GetAsync($"{serverUrl}/health");
                if (response.IsSuccessStatusCode)
                {
                    btnCheckConnection.BackColor = Color.Green;
                    AppendOutput("Server connection: OK");
                }
                else
                {
                    btnCheckConnection.BackColor = Color.Red;
                    AppendOutput("Server connection: FAILED");
                }
            }
            catch
            {
                btnCheckConnection.BackColor = Color.Red;
                AppendOutput("Server connection: FAILED");
            }
        }

        private async Task ProcessLogFile(string filePath, CancellationToken token)
        {
            AppendOutput($"Processing file: {filePath}");
            AppendOutput($"Server URL: {serverUrl}");
            AppendOutput("");

            var lines = File.ReadAllLines(filePath);
            processedCount = 0;

            foreach (var line in lines)
            {
                if (token.IsCancellationRequested) break;
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    var request = new { line = line, client_id = clientId };
                    var json = JsonSerializer.Serialize(request);
                    var content = new StringContent(json, Encoding.UTF8, "application/json");

                    var response = await client.PostAsync($"{serverUrl}/process_line", content);
                    var responseBody = await response.Content.ReadAsStringAsync();

                    if (response.IsSuccessStatusCode)
                    {
                        var result = JsonSerializer.Deserialize<ProcessResult>(responseBody);
                        processedCount++;
                        
                        // Truncate line for display
                        var displayLine = line.Length > 100 ? line.Substring(0, 100) + "..." : line;
                        AppendOutput($"Processed: {displayLine}");
                        
                        lblProcessed.Invoke((Action)(() => 
                            lblProcessed.Text = $"{processedCount} lines processed"));
                    }
                }
                catch (Exception ex)
                {
                    AppendOutput($"Error: {ex.Message}");
                }

                await Task.Delay(10); // 100 lines/second
            }

            AppendOutput($"\nCompleted! Total: {processedCount} lines processed.");
        }

        private void AppendOutput(string text)
        {
            if (txtOutput.InvokeRequired)
            {
                txtOutput.Invoke((Action)(() => AppendOutput(text)));
                return;
            }
            txtOutput.AppendText(text + Environment.NewLine);
        }

        private class ProcessResult
        {
            public string result { get; set; }
        }
    }

    /// <summary>
    /// Server Monitor form.
    /// Matches original: shows blocks, metrics with progress bars.
    /// </summary>
    public class ServerMonitorForm : Form
    {
        private ComboBox cmbClients;
        private Label lblTotalLogCount;
        private Label lblTotalBlockCount;
        private Label lblLastUpdated;
        private DataGridView dgvBlocks;
        private Button btnRefresh;
        private Button btnLoadLabels;
        
        // Metrics
        private Label lblAccuracy;
        private Label lblPrecision;
        private Label lblRecall;
        private Label lblF1Score;
        private ProgressBar pbAccuracy;
        private ProgressBar pbPrecision;
        private ProgressBar pbRecall;
        private ProgressBar pbF1Score;
        private Label lblMetricsUpdated;
        private Label lblConfusion;
        
        private static readonly HttpClient client = new HttpClient();
        private string serverUrl = "http://localhost:8000";
        private System.Windows.Forms.Timer refreshTimer;

        public ServerMonitorForm()
        {
            InitializeComponent();
            refreshTimer = new System.Windows.Forms.Timer { Interval = 5000 };
            refreshTimer.Tick += async (s, e) => await RefreshData();
            refreshTimer.Start();
            _ = RefreshData();
        }

        private void InitializeComponent()
        {
            this.Text = "Server Monitor";
            this.Size = new Size(620, 550);
            this.StartPosition = FormStartPosition.CenterScreen;

            // Clients dropdown
            cmbClients = new ComboBox
            {
                Location = new Point(10, 10),
                Size = new Size(150, 25),
                DropDownStyle = ComboBoxStyle.DropDownList
            };
            cmbClients.Items.Add("All Clients");
            cmbClients.SelectedIndex = 0;

            // Total counts
            lblTotalLogCount = new Label
            {
                Text = "Total Log Count: 0",
                Location = new Point(10, 45),
                Size = new Size(300, 25),
                Font = new Font("Segoe UI", 11F, FontStyle.Bold)
            };

            lblTotalBlockCount = new Label
            {
                Text = "Total Block Count: 0",
                Location = new Point(10, 70),
                Size = new Size(300, 25),
                Font = new Font("Segoe UI", 11F, FontStyle.Bold)
            };

            lblLastUpdated = new Label
            {
                Text = "Last Updated: -",
                Location = new Point(10, 95),
                Size = new Size(300, 20),
                ForeColor = Color.Green,
                Font = new Font("Segoe UI", 9F)
            };

            // DataGridView for blocks
            dgvBlocks = new DataGridView
            {
                Location = new Point(10, 125),
                Size = new Size(580, 150),
                AutoSizeColumnsMode = DataGridViewAutoSizeColumnsMode.Fill,
                ReadOnly = true,
                AllowUserToAddRows = false,
                SelectionMode = DataGridViewSelectionMode.FullRowSelect
            };
            dgvBlocks.Columns.Add("BlockId", "Block ID");
            dgvBlocks.Columns.Add("LogCount", "Log Count");
            dgvBlocks.Columns.Add("LastLog", "Last Log");
            dgvBlocks.Columns.Add("Status", "Status");
            dgvBlocks.Columns.Add("ClientIds", "Client IDs");

            // Refresh button
            btnRefresh = new Button
            {
                Text = "Refresh",
                Location = new Point(500, 285),
                Size = new Size(90, 30),
                BackColor = Color.FromArgb(0, 120, 215),
                ForeColor = Color.White
            };
            btnRefresh.Click += async (s, e) => await RefreshData();

            // Load Labels button
            btnLoadLabels = new Button
            {
                Text = "Load Labels",
                Location = new Point(300, 285),
                Size = new Size(90, 30)
            };
            btnLoadLabels.Click += async (s, e) => await LoadLabels();

            // Metrics section
            int metricsY = 325;
            
            lblAccuracy = new Label { Text = "Accuracy: 0.0000", Location = new Point(10, metricsY), Size = new Size(120, 20), Font = new Font("Segoe UI", 9F) };
            pbAccuracy = new ProgressBar { Location = new Point(130, metricsY), Size = new Size(450, 20), ForeColor = Color.Green };
            
            lblPrecision = new Label { Text = "Precision: 0.0000", Location = new Point(10, metricsY + 25), Size = new Size(120, 20), Font = new Font("Segoe UI", 9F) };
            pbPrecision = new ProgressBar { Location = new Point(130, metricsY + 25), Size = new Size(450, 20), ForeColor = Color.Green };
            
            lblRecall = new Label { Text = "Recall: 0.0000", Location = new Point(10, metricsY + 50), Size = new Size(120, 20), Font = new Font("Segoe UI", 9F) };
            pbRecall = new ProgressBar { Location = new Point(130, metricsY + 50), Size = new Size(450, 20), ForeColor = Color.Green };
            
            lblF1Score = new Label { Text = "F1 Score: 0.0000", Location = new Point(10, metricsY + 75), Size = new Size(120, 20), Font = new Font("Segoe UI", 9F) };
            pbF1Score = new ProgressBar { Location = new Point(130, metricsY + 75), Size = new Size(450, 20), ForeColor = Color.Green };

            lblMetricsUpdated = new Label
            {
                Text = "Last Updated: -",
                Location = new Point(10, metricsY + 100),
                Size = new Size(300, 20),
                ForeColor = Color.Green
            };

            lblConfusion = new Label
            {
                Text = "TP: 0, FP: 0, TN: 0, FN: 0",
                Location = new Point(10, metricsY + 120),
                Size = new Size(400, 20),
                Font = new Font("Segoe UI", 9F)
            };

            this.Controls.Add(cmbClients);
            this.Controls.Add(lblTotalLogCount);
            this.Controls.Add(lblTotalBlockCount);
            this.Controls.Add(lblLastUpdated);
            this.Controls.Add(dgvBlocks);
            this.Controls.Add(btnRefresh);
            this.Controls.Add(btnLoadLabels);
            this.Controls.Add(lblAccuracy);
            this.Controls.Add(pbAccuracy);
            this.Controls.Add(lblPrecision);
            this.Controls.Add(pbPrecision);
            this.Controls.Add(lblRecall);
            this.Controls.Add(pbRecall);
            this.Controls.Add(lblF1Score);
            this.Controls.Add(pbF1Score);
            this.Controls.Add(lblMetricsUpdated);
            this.Controls.Add(lblConfusion);
        }

        private async Task RefreshData()
        {
            try
            {
                // Get blocks
                var blocksResponse = await client.GetStringAsync($"{serverUrl}/blocks");
                var blocksData = JsonSerializer.Deserialize<BlocksResponse>(blocksResponse);

                dgvBlocks.Rows.Clear();
                foreach (var block in blocksData.blocks)
                {
                    dgvBlocks.Rows.Add(
                        block.block_id,
                        block.log_count,
                        block.last_log,
                        block.status,
                        string.Join(",", block.client_ids)
                    );
                }

                lblTotalLogCount.Text = $"Total Log Count: {blocksData.total_log_count:N0}";
                lblTotalBlockCount.Text = $"Total Block Count: {blocksData.total_block_count:N0}";
                lblLastUpdated.Text = $"Last Updated: {blocksData.last_updated}";

                // Get metrics
                var metricsResponse = await client.GetStringAsync($"{serverUrl}/metrics");
                var metrics = JsonSerializer.Deserialize<MetricsResponse>(metricsResponse);

                lblAccuracy.Text = $"Accuracy: {metrics.accuracy:F4}";
                pbAccuracy.Value = (int)(metrics.accuracy * 100);

                lblPrecision.Text = $"Precision: {metrics.precision:F4}";
                pbPrecision.Value = (int)(metrics.precision * 100);

                lblRecall.Text = $"Recall: {metrics.recall:F4}";
                pbRecall.Value = (int)(metrics.recall * 100);

                lblF1Score.Text = $"F1 Score: {metrics.f1_score:F4}";
                pbF1Score.Value = (int)(metrics.f1_score * 100);

                lblMetricsUpdated.Text = $"Last Updated: {metrics.last_updated}";
                lblConfusion.Text = $"TP: {metrics.tp}, FP: {metrics.fp}, TN: {metrics.tn}, FN: {metrics.fn}";
            }
            catch (Exception ex)
            {
                lblLastUpdated.Text = $"Error: {ex.Message}";
                lblLastUpdated.ForeColor = Color.Red;
            }
        }

        private async Task LoadLabels()
        {
            try
            {
                var response = await client.PostAsync($"{serverUrl}/load_labels", null);
                if (response.IsSuccessStatusCode)
                {
                    MessageBox.Show("Labels loaded successfully!", "Success", 
                        MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading labels: {ex.Message}", "Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            refreshTimer?.Stop();
            base.OnFormClosing(e);
        }

        private class BlocksResponse
        {
            public List<BlockInfo> blocks { get; set; }
            public int total_log_count { get; set; }
            public int total_block_count { get; set; }
            public string last_updated { get; set; }
        }

        private class BlockInfo
        {
            public string block_id { get; set; }
            public int log_count { get; set; }
            public string last_log { get; set; }
            public string status { get; set; }
            public List<int> client_ids { get; set; }
        }

        private class MetricsResponse
        {
            public float accuracy { get; set; }
            public float precision { get; set; }
            public float recall { get; set; }
            public float f1_score { get; set; }
            public int tp { get; set; }
            public int fp { get; set; }
            public int tn { get; set; }
            public int fn { get; set; }
            public string last_updated { get; set; }
        }
    }

    static class Program
    {
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm());
        }
    }
}
