using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Forms;

namespace NeuralNetwork1
{
    public partial class CameraCaptureForm : Form
    {
        private System.Windows.Forms.Timer captureTimer;

        [DllImport("avicap32.dll")]
        private static extern IntPtr capCreateCaptureWindowA(string lpszWindowName, int dwStyle, int x, int y, int nWidth, int nHeight, IntPtr hWndParent, int nID);

        [DllImport("user32.dll")]
        private static extern bool SendMessage(IntPtr hWnd, int wMsg, int wParam, int lParam);

        [DllImport("user32.dll")]
        private static extern bool DestroyWindow(IntPtr hWnd);

        private const int WM_USER = 0x400;
        private const int WM_CAP_DRIVER_CONNECT = WM_USER + 10;
        private const int WM_CAP_DRIVER_DISCONNECT = WM_USER + 11;
        private const int WM_CAP_SET_PREVIEW = WM_USER + 50;
        private const int WM_CAP_SET_PREVIEWRATE = WM_USER + 52;
        private const int WM_CAP_SET_SCALE = WM_USER + 53;
        private const int WM_CAP_EDIT_COPY = WM_USER + 30;

        private const int WS_CHILD = 0x40000000;
        private const int WS_VISIBLE = 0x10000000;

        private IntPtr captureWindow = IntPtr.Zero;
        private BaseNetwork neuralNetwork;
        private bool isProcessing = false;
        private bool isAutoCapture = true;
        private int threshold = 128; // Порог бинаризации по умолчанию

        public CameraCaptureForm(BaseNetwork network)
        {
            InitializeComponent();
            neuralNetwork = network;

            pictureBoxCamera.SizeMode = PictureBoxSizeMode.Zoom;
            pictureBoxProcessed.SizeMode = PictureBoxSizeMode.Zoom;

            trackBarThreshold.Value = threshold;
            lblThreshold.Text = $"Порог: {threshold}";

            InitializeCamera();
        }

        private void InitializeCamera()
        {
            try
            {
                captureWindow = capCreateCaptureWindowA("WebCam", WS_CHILD | WS_VISIBLE,
                    0, 0, pictureBoxCamera.Width, pictureBoxCamera.Height,
                    pictureBoxCamera.Handle, 0);

                if (captureWindow != IntPtr.Zero)
                {
                    if (SendMessage(captureWindow, WM_CAP_DRIVER_CONNECT, 0, 0))
                    {
                        SendMessage(captureWindow, WM_CAP_SET_SCALE, -1, 0);
                        SendMessage(captureWindow, WM_CAP_SET_PREVIEWRATE, 66, 0);
                        SendMessage(captureWindow, WM_CAP_SET_PREVIEW, -1, 0);

                        captureTimer = new System.Windows.Forms.Timer();
                        captureTimer.Interval = 150; // Немного уменьшили частоту
                        captureTimer.Tick += CaptureTimer_Tick;
                        captureTimer.Start();

                        lblStatus.Text = "Камера подключена";
                        lblStatus.ForeColor = Color.Green;
                        btnManualCapture.Text = "Остановить авто-захват";
                        isAutoCapture = true;
                    }
                    else
                    {
                        lblStatus.Text = "Не удалось подключиться к камере";
                        lblStatus.ForeColor = Color.Red;
                    }
                }
                else
                {
                    lblStatus.Text = "Не удалось создать окно захвата";
                    lblStatus.ForeColor = Color.Red;
                }
            }
            catch (Exception ex)
            {
                lblStatus.Text = $"Ошибка: {ex.Message}";
                lblStatus.ForeColor = Color.Red;
            }
        }

        private void CaptureTimer_Tick(object sender, EventArgs e)
        {
            if (isAutoCapture && !isProcessing)
            {
                CaptureAndProcessFrame();
            }
        }

        private void CaptureAndProcessFrame()
        {
            if (isProcessing || captureWindow == IntPtr.Zero)
                return;

            isProcessing = true;

            try
            {
                SendMessage(captureWindow, WM_CAP_EDIT_COPY, 0, 0);

                if (Clipboard.ContainsImage())
                {
                    using (Bitmap frame = (Bitmap)Clipboard.GetImage())
                    {
                        ProcessFrameForRecognition(frame);
                        DisplayOriginalFrame(frame);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Ошибка захвата кадра: {ex.Message}");
            }
            finally
            {
                isProcessing = false;
            }
        }

        private void DisplayOriginalFrame(Bitmap frame)
        {
            try
            {
                if (pictureBoxCamera.InvokeRequired)
                {
                    pictureBoxCamera.Invoke(new Action(() =>
                    {
                        DisplayOriginalFrameSafe(frame);
                    }));
                }
                else
                {
                    DisplayOriginalFrameSafe(frame);
                }
            }
            catch { }
        }

        private void DisplayOriginalFrameSafe(Bitmap frame)
        {
            if (pictureBoxCamera.Image != null)
            {
                pictureBoxCamera.Image.Dispose();
            }

            Bitmap displayImage = new Bitmap(pictureBoxCamera.Width, pictureBoxCamera.Height);
            using (Graphics g = Graphics.FromImage(displayImage))
            {
                g.Clear(Color.Black);

                float scale = Math.Min(
                    (float)pictureBoxCamera.Width / frame.Width,
                    (float)pictureBoxCamera.Height / frame.Height);

                int newWidth = (int)(frame.Width * scale);
                int newHeight = (int)(frame.Height * scale);
                int x = (pictureBoxCamera.Width - newWidth) / 2;
                int y = (pictureBoxCamera.Height - newHeight) / 2;

                g.DrawImage(frame, x, y, newWidth, newHeight);
            }

            pictureBoxCamera.Image = displayImage;
        }

        private void ProcessFrameForRecognition(Bitmap frame)
        {
            try
            {
                Bitmap processed = PreprocessImage(frame);

                double[] inputs = ConvertBitmapToInputVector(processed);

                Sample sample = new Sample((Bitmap)processed.Clone(), inputs, 10, Letter.undef);

                Letter predictedClass = neuralNetwork.Predict(sample);

                if (lblResult.InvokeRequired)
                {
                    lblResult.Invoke(new Action(() =>
                    {
                        UpdateResultUI(predictedClass, sample.Output, processed);
                    }));
                }
                else
                {
                    UpdateResultUI(predictedClass, sample.Output, processed);
                }

                processed.Dispose();
            }
            catch (Exception ex)
            {
                if (lblResult.InvokeRequired)
                {
                    lblResult.Invoke(new Action(() =>
                    {
                        lblResult.Text = $"Ошибка: {ex.Message}";
                        lblResult.ForeColor = Color.Red;
                    }));
                }
            }
        }

        private void UpdateResultUI(Letter predictedClass, double[] outputs, Bitmap processedImage)
        {
            lblResult.Text = $"Распознано: {predictedClass}";
            lblResult.ForeColor = Color.Green;

            if (pictureBoxProcessed.Image != null)
            {
                pictureBoxProcessed.Image.Dispose();
            }

            // Создаем изображение для отображения - УВЕЛИЧЕННОЕ для лучшей видимости
            Bitmap displayProcessed = new Bitmap(pictureBoxProcessed.Width, pictureBoxProcessed.Height);
            using (Graphics g = Graphics.FromImage(displayProcessed))
            {
                g.Clear(Color.White);

                // Увеличиваем в 10 раз для отображения
                int scale = 10;
                int displaySize = 28 * scale;
                int x = (pictureBoxProcessed.Width - displaySize) / 2;
                int y = (pictureBoxProcessed.Height - displaySize) / 2;

                // Создаем увеличенную версию
                Bitmap enlarged = new Bitmap(displaySize, displaySize);
                using (Graphics g2 = Graphics.FromImage(enlarged))
                {
                    g2.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                    g2.DrawImage(processedImage, 0, 0, displaySize, displaySize);
                }

                // Отображаем пиксели как квадратики
                g.DrawImage(enlarged, x, y, displaySize, displaySize);
                enlarged.Dispose();
            }

            pictureBoxProcessed.Image = displayProcessed;

            DisplayConfidence(outputs);
        }

        private void DisplayConfidence(double[] outputs)
        {
            string confidenceText = "Уверенность:\n";

            var predictions = new System.Collections.Generic.List<(Letter letter, double confidence)>();
            for (int i = 0; i < outputs.Length; i++)
            {
                predictions.Add(((Letter)i, outputs[i]));
            }

            predictions.Sort((a, b) => b.confidence.CompareTo(a.confidence));

            for (int i = 0; i < Math.Min(3, predictions.Count); i++)
            {
                confidenceText += $"{i + 1}. {predictions[i].letter}: {predictions[i].confidence:F3}\n";
            }

            lblConfidence.Text = confidenceText;
        }

        private Bitmap PreprocessImage(Bitmap original)
        {
            // 1. Обрезаем центральную часть
            int size = Math.Min(original.Width, original.Height);
            Rectangle cropRect = new Rectangle(
                (original.Width - size) / 2,
                (original.Height - size) / 2,
                size,
                size);

            Bitmap cropped = new Bitmap(size, size);
            using (Graphics g = Graphics.FromImage(cropped))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(original, new Rectangle(0, 0, size, size), cropRect, GraphicsUnit.Pixel);
            }

            // 2. Масштабируем до 28x28
            Bitmap resized = new Bitmap(28, 28);
            using (Graphics g = Graphics.FromImage(resized))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

                float scale = Math.Min(28f / size, 28f / size);
                int newWidth = (int)(size * scale);
                int newHeight = (int)(size * scale);
                int x = (28 - newWidth) / 2;
                int y = (28 - newHeight) / 2;

                g.DrawImage(cropped, x, y, newWidth, newHeight);
            }

            // 3. Преобразуем в градации серого
            Bitmap grayscale = new Bitmap(28, 28);

            // ВАЖНО: Вычисляем среднюю яркость КАРТИНКИ, а не фона
            long totalBrightness = 0;
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    Color pixel = resized.GetPixel(x, y);
                    // Преобразуем в градации серого
                    int grayValue = (int)(pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114);
                    grayscale.SetPixel(x, y, Color.FromArgb(grayValue, grayValue, grayValue));
                    totalBrightness += grayValue;
                }
            }

            // Используем фиксированный порог или адаптивный
            int currentThreshold = threshold;
            if (chkAutoThreshold.Checked)
            {
                int averageBrightness = (int)(totalBrightness / (28 * 28));
                currentThreshold = averageBrightness;
            }

            // 4. Бинаризация с НЕПРАВИЛЬНЫМИ цветами (инверсия)
            Bitmap binary = new Bitmap(28, 28);

            // ВАЖНО: Нейросеть ожидает ЧЕРНЫЕ буквы на БЕЛОМ фоне
            // Но камера видит БЕЛЫЙ лист с ЧЕРНОЙ буквой
            // Значит нам нужно инвертировать!
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    Color pixel = grayscale.GetPixel(x, y);
                    int grayValue = pixel.R; // В градациях серого R=G=B

                    // Если пиксель ТЕМНЫЙ (меньше порога) - это БУКВА
                    // Буква должна быть ЧЕРНОЙ для нейросети = 1.0
                    // Но на отображении мы хотим видеть черную букву на белом
                    if (grayValue < currentThreshold)
                    {
                        // Это буква - делаем ЧЕРНЫЙ (0)
                        binary.SetPixel(x, y, Color.FromArgb(0, 0, 0));
                    }
                    else
                    {
                        // Это фон - делаем БЕЛЫЙ (255)
                        binary.SetPixel(x, y, Color.FromArgb(255, 255, 255));
                    }
                }
            }

            cropped.Dispose();
            resized.Dispose();
            grayscale.Dispose();

            return binary;
        }

        private double[] ConvertBitmapToInputVector(Bitmap bitmap)
        {
            double[] inputs = new double[784];

            int index = 0;
            for (int y = 0; y < 28; y++)
            {
                for (int x = 0; x < 28; x++)
                {
                    Color pixel = bitmap.GetPixel(x, y);

                    // ВАЖНО: Нейросеть обучалась так:
                    // - Черный пиксель (0) = буква = 1.0
                    // - Белый пиксель (255) = фон = 0.0

                    // В нашем binary изображении:
                    // - Черный (0,0,0) = буква
                    // - Белый (255,255,255) = фон

                    // Поэтому преобразуем:
                    inputs[index++] = pixel.R == 0 ? 1.0 : 0.0;
                }
            }

            return inputs;
        }

        private void btnManualCapture_Click(object sender, EventArgs e)
        {
            try
            {
                isAutoCapture = !isAutoCapture;

                if (isAutoCapture)
                {
                    btnManualCapture.Text = "Остановить авто-захват";
                    lblStatus.Text = "Авто-захват включен";
                    lblStatus.ForeColor = Color.Green;
                }
                else
                {
                    btnManualCapture.Text = "Включить авто-захват";
                    lblStatus.Text = "Авто-захват отключен";
                    lblStatus.ForeColor = Color.Orange;
                    CaptureAndProcessFrame();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка: {ex.Message}", "Ошибка", MessageBoxButtons.OK);
            }
        }

        private void btnSaveImage_Click(object sender, EventArgs e)
        {
            if (pictureBoxProcessed.Image != null)
            {
                SaveFileDialog saveDialog = new SaveFileDialog();
                saveDialog.Filter = "PNG Image|*.png|JPEG Image|*.jpg|BMP Image|*.bmp";
                saveDialog.Title = "Сохранить обработанное изображение";
                saveDialog.FileName = $"letter_{DateTime.Now:yyyyMMdd_HHmmss}";

                if (saveDialog.ShowDialog() == DialogResult.OK)
                {
                    pictureBoxProcessed.Image.Save(saveDialog.FileName, GetImageFormat(saveDialog.FilterIndex));
                    MessageBox.Show("Изображение сохранено!", "Сохранение", MessageBoxButtons.OK);
                }
            }
            else
            {
                MessageBox.Show("Нет изображения для сохранения!", "Предупреждение", MessageBoxButtons.OK);
            }
        }

        private ImageFormat GetImageFormat(int filterIndex)
        {
            switch (filterIndex)
            {
                case 1: return ImageFormat.Png;
                case 2: return ImageFormat.Jpeg;
                case 3: return ImageFormat.Bmp;
                default: return ImageFormat.Png;
            }
        }

        private void CameraCaptureForm_FormClosing(object sender, FormClosingEventArgs e)
        {
            StopCamera();
        }

        private void StopCamera()
        {
            if (captureTimer != null)
            {
                captureTimer.Stop();
                captureTimer.Dispose();
                captureTimer = null;
            }

            if (captureWindow != IntPtr.Zero)
            {
                try
                {
                    SendMessage(captureWindow, WM_CAP_DRIVER_DISCONNECT, 0, 0);
                    DestroyWindow(captureWindow);
                }
                catch { }
                captureWindow = IntPtr.Zero;
            }

            if (pictureBoxCamera.Image != null)
            {
                pictureBoxCamera.Image.Dispose();
                pictureBoxCamera.Image = null;
            }

            if (pictureBoxProcessed.Image != null)
            {
                pictureBoxProcessed.Image.Dispose();
                pictureBoxProcessed.Image = null;
            }
        }

        private void CameraCaptureForm_Load(object sender, EventArgs e)
        {
            // Уже инициализировано в конструкторе
        }

        private void btnClose_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void btnSettings_Click(object sender, EventArgs e)
        {
            // Пропускаем настройки камеры для простоты
            MessageBox.Show("Используйте настройки освещения и положение камеры для лучшего распознавания.", "Совет", MessageBoxButtons.OK);
        }

        private void btnSingleCapture_Click(object sender, EventArgs e)
        {
            if (!isAutoCapture)
            {
                CaptureAndProcessFrame();
            }
        }

        private void trackBarThreshold_Scroll(object sender, EventArgs e)
        {
            threshold = trackBarThreshold.Value;
            lblThreshold.Text = $"Порог: {threshold}";
        }

        private void chkAutoThreshold_CheckedChanged(object sender, EventArgs e)
        {
            trackBarThreshold.Enabled = !chkAutoThreshold.Checked;
            if (chkAutoThreshold.Checked)
            {
                lblStatus.Text = "Автопорог включен";
            }
            else
            {
                lblStatus.Text = $"Ручной порог: {threshold}";
            }
        }

        #region Windows Form Designer generated code

        private System.ComponentModel.IContainer components = null;
        private PictureBox pictureBoxCamera;
        private PictureBox pictureBoxProcessed;
        private Button btnManualCapture;
        private Button btnClose;
        private Label lblResult;
        private Label label1;
        private Label label2;
        private Label lblStatus;
        private Label lblConfidence;
        private Button btnSaveImage;
        private Button btnSettings;
        private Button btnSingleCapture;
        private TrackBar trackBarThreshold;
        private Label lblThreshold;
        private CheckBox chkAutoThreshold;

        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        private void InitializeComponent()
        {
            this.pictureBoxCamera = new System.Windows.Forms.PictureBox();
            this.pictureBoxProcessed = new System.Windows.Forms.PictureBox();
            this.btnManualCapture = new System.Windows.Forms.Button();
            this.btnClose = new System.Windows.Forms.Button();
            this.lblResult = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.lblStatus = new System.Windows.Forms.Label();
            this.lblConfidence = new System.Windows.Forms.Label();
            this.btnSaveImage = new System.Windows.Forms.Button();
            this.btnSettings = new System.Windows.Forms.Button();
            this.btnSingleCapture = new System.Windows.Forms.Button();
            this.trackBarThreshold = new System.Windows.Forms.TrackBar();
            this.lblThreshold = new System.Windows.Forms.Label();
            this.chkAutoThreshold = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCamera)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxProcessed)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarThreshold)).BeginInit();
            this.SuspendLayout();

            // pictureBoxCamera
            this.pictureBoxCamera.BackColor = System.Drawing.Color.Black;
            this.pictureBoxCamera.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pictureBoxCamera.Location = new System.Drawing.Point(12, 35);
            this.pictureBoxCamera.Name = "pictureBoxCamera";
            this.pictureBoxCamera.Size = new System.Drawing.Size(320, 240);
            this.pictureBoxCamera.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBoxCamera.TabIndex = 0;
            this.pictureBoxCamera.TabStop = false;

            // pictureBoxProcessed
            this.pictureBoxProcessed.BackColor = System.Drawing.Color.White;
            this.pictureBoxProcessed.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pictureBoxProcessed.Location = new System.Drawing.Point(350, 35);
            this.pictureBoxProcessed.Name = "pictureBoxProcessed";
            this.pictureBoxProcessed.Size = new System.Drawing.Size(280, 280);
            this.pictureBoxProcessed.SizeMode = System.Windows.Forms.PictureBoxSizeMode.Zoom;
            this.pictureBoxProcessed.TabIndex = 1;
            this.pictureBoxProcessed.TabStop = false;

            // btnManualCapture
            this.btnManualCapture.Location = new System.Drawing.Point(12, 300);
            this.btnManualCapture.Name = "btnManualCapture";
            this.btnManualCapture.Size = new System.Drawing.Size(150, 40);
            this.btnManualCapture.TabIndex = 2;
            this.btnManualCapture.Text = "Остановить авто-захват";
            this.btnManualCapture.UseVisualStyleBackColor = true;
            this.btnManualCapture.Click += new System.EventHandler(this.btnManualCapture_Click);

            // btnClose
            this.btnClose.Location = new System.Drawing.Point(12, 390);
            this.btnClose.Name = "btnClose";
            this.btnClose.Size = new System.Drawing.Size(150, 40);
            this.btnClose.TabIndex = 3;
            this.btnClose.Text = "Закрыть";
            this.btnClose.UseVisualStyleBackColor = true;
            this.btnClose.Click += new System.EventHandler(this.btnClose_Click);

            // lblResult
            this.lblResult.AutoSize = true;
            this.lblResult.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.lblResult.Location = new System.Drawing.Point(350, 320);
            this.lblResult.Name = "lblResult";
            this.lblResult.Size = new System.Drawing.Size(148, 20);
            this.lblResult.TabIndex = 4;
            this.lblResult.Text = "Результат: -----";

            // label1
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 15);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(139, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Веб-камера (режим онлайн)";

            // label2
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(350, 15);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(154, 13);
            this.label2.TabIndex = 6;
            this.label2.Text = "Обработанное (28x28)";

            // lblStatus
            this.lblStatus.AutoSize = true;
            this.lblStatus.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(204)));
            this.lblStatus.Location = new System.Drawing.Point(12, 280);
            this.lblStatus.Name = "lblStatus";
            this.lblStatus.Size = new System.Drawing.Size(99, 13);
            this.lblStatus.TabIndex = 7;
            this.lblStatus.Text = "Статус: Загрузка";

            // lblConfidence
            this.lblConfidence.AutoSize = true;
            this.lblConfidence.Location = new System.Drawing.Point(350, 350);
            this.lblConfidence.Name = "lblConfidence";
            this.lblConfidence.Size = new System.Drawing.Size(280, 52);
            this.lblConfidence.TabIndex = 8;
            this.lblConfidence.Text = "Уверенность:\n1. ---: ---\n2. ---: ---\n3. ---: ---";

            // btnSaveImage
            this.btnSaveImage.Location = new System.Drawing.Point(170, 300);
            this.btnSaveImage.Name = "btnSaveImage";
            this.btnSaveImage.Size = new System.Drawing.Size(130, 40);
            this.btnSaveImage.TabIndex = 9;
            this.btnSaveImage.Text = "Сохранить";
            this.btnSaveImage.UseVisualStyleBackColor = true;
            this.btnSaveImage.Click += new System.EventHandler(this.btnSaveImage_Click);

            // btnSettings
            this.btnSettings.Location = new System.Drawing.Point(170, 390);
            this.btnSettings.Name = "btnSettings";
            this.btnSettings.Size = new System.Drawing.Size(130, 40);
            this.btnSettings.TabIndex = 10;
            this.btnSettings.Text = "Советы";
            this.btnSettings.UseVisualStyleBackColor = true;
            this.btnSettings.Click += new System.EventHandler(this.btnSettings_Click);

            // btnSingleCapture
            this.btnSingleCapture.Location = new System.Drawing.Point(12, 345);
            this.btnSingleCapture.Name = "btnSingleCapture";
            this.btnSingleCapture.Size = new System.Drawing.Size(150, 40);
            this.btnSingleCapture.TabIndex = 11;
            this.btnSingleCapture.Text = "Сделать снимок";
            this.btnSingleCapture.UseVisualStyleBackColor = true;
            this.btnSingleCapture.Click += new System.EventHandler(this.btnSingleCapture_Click);

            // trackBarThreshold
            this.trackBarThreshold.Location = new System.Drawing.Point(170, 345);
            this.trackBarThreshold.Maximum = 255;
            this.trackBarThreshold.Name = "trackBarThreshold";
            this.trackBarThreshold.Size = new System.Drawing.Size(130, 45);
            this.trackBarThreshold.TabIndex = 12;
            this.trackBarThreshold.Value = 128;
            this.trackBarThreshold.Scroll += new System.EventHandler(this.trackBarThreshold_Scroll);

            // lblThreshold
            this.lblThreshold.AutoSize = true;
            this.lblThreshold.Location = new System.Drawing.Point(170, 320);
            this.lblThreshold.Name = "lblThreshold";
            this.lblThreshold.Size = new System.Drawing.Size(58, 13);
            this.lblThreshold.TabIndex = 13;
            this.lblThreshold.Text = "Порог: 128";

            // chkAutoThreshold
            this.chkAutoThreshold.AutoSize = true;
            this.chkAutoThreshold.Checked = true;
            this.chkAutoThreshold.Location = new System.Drawing.Point(310, 345);
            this.chkAutoThreshold.Name = "chkAutoThreshold";
            this.chkAutoThreshold.Size = new System.Drawing.Size(89, 17);
            this.chkAutoThreshold.TabIndex = 14;
            this.chkAutoThreshold.Text = "Автопорог";
            this.chkAutoThreshold.UseVisualStyleBackColor = true;
            this.chkAutoThreshold.CheckedChanged += new System.EventHandler(this.chkAutoThreshold_CheckedChanged);

            // CameraCaptureForm
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(644, 440);
            this.Controls.Add(this.chkAutoThreshold);
            this.Controls.Add(this.lblThreshold);
            this.Controls.Add(this.trackBarThreshold);
            this.Controls.Add(this.btnSingleCapture);
            this.Controls.Add(this.btnSettings);
            this.Controls.Add(this.btnSaveImage);
            this.Controls.Add(this.lblConfidence);
            this.Controls.Add(this.lblStatus);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblResult);
            this.Controls.Add(this.btnClose);
            this.Controls.Add(this.btnManualCapture);
            this.Controls.Add(this.pictureBoxProcessed);
            this.Controls.Add(this.pictureBoxCamera);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedDialog;
            this.MaximizeBox = false;
            this.Name = "CameraCaptureForm";
            this.Text = "Распознавание греческих букв с веб-камеры";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.CameraCaptureForm_FormClosing);
            this.Load += new System.EventHandler(this.CameraCaptureForm_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxCamera)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBoxProcessed)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.trackBarThreshold)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();
        }

        #endregion
    }
}