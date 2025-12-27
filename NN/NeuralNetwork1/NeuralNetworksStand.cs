using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Text.RegularExpressions;
using MathNet.Numerics;
using Accord.Neuro;

namespace NeuralNetwork1
{
    public partial class NeuralNetworksStand : Form
    {
        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        GenerateImage generator = new GenerateImage();

        //  Создаём новую обучающую выборку
        SamplesSet samples = new SamplesSet();

        public Letter GetClassNum(string fileName)
        {
            switch ((fileName.Split('\\')[1]).Split('_')[0])
            {
                case "alpha":
                    return Letter.alpha;
                case "beta":
                    return Letter.beta;
                case "gamma":
                    return Letter.gamma;
                case "delta":
                    return Letter.delta;
                case "epsilon":
                    return Letter.epsilon;
                case "pi":
                    return Letter.pi;
                case "omega":
                    return Letter.omega;
                case "teta":
                    return Letter.teta;
                case "psi":
                    return Letter.psi;
                case "nu":
                    return Letter.nu;
                default:
                    return Letter.undef;
            }
        }

        /// <summary>
        /// Текущая выбранная через селектор нейросеть
        /// </summary>
        public BaseNetwork Net
        {
            get
            {
                var selectedItem = (string)netTypeBox.SelectedItem;
                if (!networksCache.ContainsKey(selectedItem))
                    networksCache.Add(selectedItem, CreateNetwork(selectedItem));

                return networksCache[selectedItem];
            }
        }

        private readonly Dictionary<string, Func<int[], BaseNetwork>> networksFabric;
        private Dictionary<string, BaseNetwork> networksCache = new Dictionary<string, BaseNetwork>();

        /// <summary>
        /// Конструктор формы стенда для работы с сетями
        /// </summary>
        /// <param name="networksFabric">Словарь функций, создающих сети с заданной структурой</param>
        public NeuralNetworksStand(Dictionary<string, Func<int[], BaseNetwork>> networksFabric)
        {
            InitializeComponent();

            string[] imageFiles = Directory.GetFiles("imgs", "*.*", SearchOption.TopDirectoryOnly).ToArray();
            foreach (string file in imageFiles)
            {
                Console.WriteLine(file);

                // Загружаем изображение (уже 28x28 и черно-белое)
                Bitmap bmp = new Bitmap(file);

                // Проверяем размер
                if (bmp.Width != 28 || bmp.Height != 28)
                {
                    MessageBox.Show($"Изображение {file} имеет размер {bmp.Width}x{bmp.Height}, а должен быть 28x28!", "Ошибка", MessageBoxButtons.OK);
                    continue;
                }

                // Создаем входной вектор размером 784 (все пиксели 28x28)
                double[] inputs = new double[784];

                // Преобразуем каждый пиксель в значение 0 (белый) или 1 (черный)
                int inputIndex = 0;
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        Color pixel = bmp.GetPixel(x, y);
                        // Для черно-белых изображений: черный = 1, белый = 0
                        // Проверяем, является ли пиксель черным (все каналы близки к 0)
                        inputs[inputIndex++] = (pixel.R < 128 && pixel.G < 128 && pixel.B < 128) ? 1.0 : 0.0;
                    }
                }

                // Создаем образ с 10 классами (по количеству enum Letter)
                samples.AddSample(new Sample(bmp, inputs, 10, GetClassNum(file)));
            }


            samples.Shuffle();
            this.Height = 580;
            this.Width = 850;
            this.networksFabric = networksFabric;
            netTypeBox.Items.AddRange(this.networksFabric.Keys.Select(s => (object)s).ToArray());
            netTypeBox.SelectedIndex = 0;
            generator.FigureCount = (int)classCounter.Value;

            // Устанавливаем структуру сети по умолчанию для 28x28 изображений
            netStructureBox.Text = "784;32;10";

            button3_Click(this, null);
            pictureBox1.Image = Properties.Resources.Title;

            // Добавляем обработчик для кнопки камеры
            btnCamera.Click += BtnCamera_Click;
        }

        // Обработчик для кнопки "Распознать с камеры"
        private void BtnCamera_Click(object sender, EventArgs e)
        {
            if (Net == null)
            {
                MessageBox.Show("Сначала создайте и обучите сеть!", "Ошибка", MessageBoxButtons.OK);
                return;
            }

            try
            {
                CameraCaptureForm cameraForm = new CameraCaptureForm(Net);
                cameraForm.ShowDialog();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Не удалось открыть камеру: {ex.Message}\n\n" +
                               "Возможные причины:\n" +
                               "1. Веб-камера не подключена\n" +
                               "2. Камера занята другим приложением\n" +
                               "3. Нет драйверов для камеры\n" +
                               "4. Система не поддерживает старый Windows API",
                               "Ошибка", MessageBoxButtons.OK);
            }
        }

        public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
        {
            if (progressBar1.InvokeRequired)
            {
                progressBar1.Invoke(new TrainProgressHandler(UpdateLearningInfo), progress, error, elapsedTime);
                return;
            }

            StatusLabel.Text = "Ошибка: " + error;
            int progressPercent = (int)Math.Round(progress * 100);
            progressPercent = Math.Min(100, Math.Max(0, progressPercent));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = progressPercent;
        }


        private void set_result(Sample figure)
        {
            label1.ForeColor = figure.Correct() ? Color.Green : Color.Red;

            label1.Text = "Распознано : " + figure.recognizedClass;

            label8.Text = string.Join("\n", figure.Output.Select(d => d.ToString(CultureInfo.InvariantCulture)));

            // Масштабируем изображение для отображения
            Bitmap scaledBitmap = new Bitmap(200, 200);
            using (Graphics g = Graphics.FromImage(scaledBitmap))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                g.DrawImage(figure.bitmap, 0, 0, 200, 200);
            }

            pictureBox1.Image = scaledBitmap;
            pictureBox1.Invalidate();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            if (samples.Count == 0)
            {
                MessageBox.Show("Нет загруженных изображений!", "Ошибка", MessageBoxButtons.OK);
                return;
            }

            Random r = new Random();
            Sample fig = samples[r.Next(samples.Count)];

            Net.Predict(fig);
            set_result(fig);
        }

        private async Task<double> train_networkAsync(int training_size, int epoches, double acceptable_error,
    bool parallel = true)
        {
            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            trainOneButton.Enabled = false;

            try
            {
                //  Обучение запускаем асинхронно, чтобы не блокировать форму
                var curNet = Net;

                // Сбрасываем сеть StudentNetwork перед обучением
                if (curNet is StudentNetwork studentNet)
                {
                    studentNet.ResetNetwork();
                }

                double f = await Task.Run(() => curNet.TrainOnDataSet(samples, epoches, acceptable_error, parallel));

                label1.Text = "Щелкните на картинку для теста нового образа";
                label1.ForeColor = Color.Green;
                groupBox1.Enabled = true;
                pictureBox1.Enabled = true;
                trainOneButton.Enabled = true;
                StatusLabel.Text = "Ошибка: " + f.ToString("F6");
                StatusLabel.ForeColor = Color.Green;
                return f;
            }
            catch (Exception e)
            {
                label1.Text = $"Исключение: {e.Message}";
                StatusLabel.Text = "Ошибка обучения!";
                StatusLabel.ForeColor = Color.Red;
            }
            finally
            {
                groupBox1.Enabled = true;
                pictureBox1.Enabled = true;
                trainOneButton.Enabled = true;
            }

            return 0;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (samples.Count == 0)
            {
                MessageBox.Show("Нет данных для обучения!", "Ошибка", MessageBoxButtons.OK);
                return;
            }

#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync((int)TrainingSizeCounter.Value, (int)EpochesCounter.Value,
                (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed

            ExportNetwork(netTypeBox.SelectedIndex == 1);
        }

        public void ExportNetwork(bool student)
        {
            if (student)
            {
                (Net as StudentNetwork).ExportNetwork();
            }
            else
            {
                using (FileStream fs = new FileStream("networks\\libNet.bin", FileMode.Create))
                {
                    var formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    formatter.Serialize(fs, (Net as AccordNet).network);
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (samples.Count == 0)
            {
                MessageBox.Show("Нет данных для тестирования!", "Ошибка", MessageBoxButtons.OK);
                return;
            }

            Enabled = false;

            double accuracy = samples.TestNeuralNetwork(Net);

            StatusLabel.Text = $"Точность на тестовой выборке : {accuracy * 100,5:F2}%";
            StatusLabel.ForeColor = accuracy * 100 >= AccuracyCounter.Value ? Color.Green : Color.Red;

            Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            try
            {
                //  Проверяем корректность задания структуры сети
                int[] structure = CurrentNetworkStructure();
                if (structure.Length < 2)
                {
                    MessageBox.Show("В сети должно быть как минимум 2 слоя!", "Ошибка", MessageBoxButtons.OK);
                    return;
                }

                if (structure[0] != 784)  // 28x28 = 784 пикселей
                {
                    MessageBox.Show($"Первый слой должен содержать 784 нейрона (28x28 пикселей). Указано: {structure[0]}", "Ошибка", MessageBoxButtons.OK);
                    return;
                }

                if (structure[structure.Length - 1] != 10)
                {
                    MessageBox.Show($"Последний слой должен содержать 10 нейронов (10 классов). Указано: {structure[structure.Length - 1]}", "Ошибка", MessageBoxButtons.OK);
                    return;
                }

                // Чистим старые подписки сетей
                foreach (var network in networksCache.Values)
                    network.TrainProgress -= UpdateLearningInfo;

                // Очищаем кэш сетей
                networksCache.Clear();

                // Обновляем текущую сеть
                var selectedItem = (string)netTypeBox.SelectedItem;
                networksCache[selectedItem] = CreateNetwork(selectedItem);
            }
            catch (FormatException)
            {
                MessageBox.Show("Некорректный формат структуры сети. Используйте формат: 784;32;10", "Ошибка", MessageBoxButtons.OK);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Ошибка при создании сети: {ex.Message}", "Ошибка", MessageBoxButtons.OK);
            }
        }

        private int[] CurrentNetworkStructure()
        {
            try
            {
                return netStructureBox.Text.Split(';')
                    .Select(s => s.Trim())
                    .Where(s => !string.IsNullOrEmpty(s))
                    .Select(int.Parse)
                    .ToArray();
            }
            catch
            {
                // Возвращаем структуру по умолчанию при ошибке
                return new int[] { 784, 32, 10 };
            }
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.FigureCount = (int)classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            if (!int.TryParse(vals.Last(), out _)) return;
            vals[vals.Length - 1] = classCounter.Value.ToString();
            netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (Net == null || samples.Count == 0) return;
            Random r = new Random();
            Sample fig = samples[r.Next(samples.Count)];
            pictureBox1.Invalidate();
            Net.Train(fig, 0.00005, parallelCheckBox.Checked);
            set_result(fig);
        }

        private BaseNetwork CreateNetwork(string networkName)
        {
            var network = networksFabric[networkName](CurrentNetworkStructure());
            network.TrainProgress += UpdateLearningInfo;
            return network;
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке такого же размера";
        }
    }
}