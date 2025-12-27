using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private int[] structure;
        private double[][] neurons;
        private double[][][] weights;
        private double[][] biases;
        private double learningRate = 0.15;
        private Random random;

        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            InitializeNetwork(structure);
        }

        private void InitializeNetwork(int[] structure)
        {
            this.structure = structure;
            neurons = new double[structure.Length][];
            weights = new double[structure.Length - 1][][];
            biases = new double[structure.Length - 1][];
            random = new Random();

            // Инициализация нейронов
            for (int i = 0; i < structure.Length; i++)
                neurons[i] = new double[structure[i]];

            // Инициализация весов и смещений
            for (int i = 0; i < structure.Length - 1; i++)
            {
                int currentLayerSize = structure[i];
                int nextLayerSize = structure[i + 1];

                weights[i] = new double[nextLayerSize][];
                biases[i] = new double[nextLayerSize];

                for (int j = 0; j < nextLayerSize; j++)
                {
                    weights[i][j] = new double[currentLayerSize];

                    // Инициализация весов по алгоритму Xavier/Glorot
                    double range = Math.Sqrt(6.0 / (currentLayerSize + nextLayerSize));
                    for (int k = 0; k < currentLayerSize; k++)
                        weights[i][j][k] = random.NextDouble() * 2 * range - range;

                    biases[i][j] = random.NextDouble() * 0.1 - 0.05; // небольшие значения
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            double error;
            int iters = 0;

            do
            {
                error = TrainSample(sample.input, sample.Output);
                iters++;
                // Защита от бесконечного цикла
                if (iters > 10000)
                    break;
            } while (error > acceptableError);

            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double totalError = double.PositiveInfinity;

            stopWatch.Restart();

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                totalError = 0;
                int correctCount = 0;

                // Перемешиваем данные в каждой эпохе
                samplesSet.Shuffle();

                foreach (Sample sample in samplesSet.samples)
                {
                    totalError += TrainSample(sample.input, sample.Output);

                    // Проверяем правильность классификации
                    var prediction = Compute(sample.input);
                    int predictedClass = Array.IndexOf(prediction, prediction.Max());
                    if (predictedClass == (int)sample.actualClass)
                        correctCount++;
                }

                totalError /= samplesSet.Count;
                double accuracy = (double)correctCount / samplesSet.Count;

                if (totalError <= acceptableError)
                    break;

                // Выводим прогресс с точностью
                OnTrainProgress((double)(epoch + 1) / epochsCount, totalError, stopWatch.Elapsed);

                // Адаптивное уменьшение learning rate
                if (epoch > 0 && epoch % 10 == 0)
                {
                    learningRate *= 0.95;
                }

                // Защита от NaN/Infinity
                if (double.IsNaN(totalError) || double.IsInfinity(totalError))
                {
                    // Сбрасываем сеть при возникновении численных проблем
                    InitializeNetwork(structure);
                    learningRate = 0.15;
                }
            }

            OnTrainProgress(1.0, totalError, stopWatch.Elapsed);

            stopWatch.Stop();

            return totalError;
        }

        protected override double[] Compute(double[] input)
        {
            // Копируем входные данные в первый слой
            Array.Copy(input, neurons[0], input.Length);

            // Прямое распространение через все слои
            for (int layer = 0; layer < weights.Length; layer++)
            {
                Parallel.For(0, neurons[layer + 1].Length, neuron =>
                {
                    double sum = biases[layer][neuron];

                    // Векторизованное вычисление (для производительности)
                    for (int prevNeuron = 0; prevNeuron < neurons[layer].Length; prevNeuron++)
                    {
                        sum += neurons[layer][prevNeuron] * weights[layer][neuron][prevNeuron];
                    }

                    // Функция активации
                    neurons[layer + 1][neuron] = Sigmoid(sum);
                });
            }

            return neurons[neurons.Length - 1];
        }

        private double TrainSample(double[] inputs, double[] expectedOutputs)
        {
            // Прямое распространение
            var outputs = Compute(inputs);

            // Массивы для хранения ошибок
            double[][] errors = new double[structure.Length][];
            for (int i = 0; i < structure.Length; i++)
                errors[i] = new double[structure[i]];

            // Ошибка выходного слоя
            double totalError = 0;
            for (int i = 0; i < expectedOutputs.Length; i++)
            {
                errors[errors.Length - 1][i] = expectedOutputs[i] - outputs[i];
                totalError += Math.Pow(errors[errors.Length - 1][i], 2);
            }
            totalError /= 2;

            // Обратное распространение ошибки
            for (int layer = weights.Length - 1; layer >= 0; layer--)
            {
                Parallel.For(0, weights[layer].Length, neuron =>
                {
                    // Вычисляем дельту для текущего нейрона
                    double delta;
                    if (layer == weights.Length - 1)
                    {
                        // Выходной слой
                        delta = errors[layer + 1][neuron] * SigmoidDerivative(neurons[layer + 1][neuron]);
                    }
                    else
                    {
                        // Скрытый слой
                        delta = 0;
                        for (int nextNeuron = 0; nextNeuron < weights[layer + 1].Length; nextNeuron++)
                        {
                            delta += errors[layer + 2][nextNeuron] * weights[layer + 1][nextNeuron][neuron];
                        }
                        delta *= SigmoidDerivative(neurons[layer + 1][neuron]);
                    }

                    // Сохраняем ошибку для предыдущего слоя
                    errors[layer + 1][neuron] = delta;

                    // Обновляем веса, идущие в текущий нейрон
                    for (int prevNeuron = 0; prevNeuron < neurons[layer].Length; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] += learningRate * delta * neurons[layer][prevNeuron];
                    }

                    // Обновляем смещение
                    biases[layer][neuron] += learningRate * delta;
                });
            }

            return totalError;
        }

        private double Sigmoid(double x)
        {
            // Защита от переполнения
            if (x < -45) return 0;
            if (x > 45) return 1;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            // x уже должно быть значением активации (выходом sigmoid)
            return x * (1.0 - x);
        }

        public void ResetNetwork()
        {
            InitializeNetwork(structure);
            learningRate = 0.15;
        }

        public void ExportNetwork()
        {
            Directory.CreateDirectory("networks");

            using (var writer = new BinaryWriter(File.Open("networks\\studNet.bin", FileMode.Create)))
            {
                // Сохраняем структуру
                writer.Write(structure.Length);
                foreach (var layerSize in structure)
                    writer.Write(layerSize);

                // Сохраняем веса
                writer.Write(weights.Length);
                foreach (var layerWeights in weights)
                {
                    writer.Write(layerWeights.Length);
                    foreach (var neuronWeights in layerWeights)
                    {
                        writer.Write(neuronWeights.Length);
                        foreach (var weight in neuronWeights)
                            writer.Write(weight);
                    }
                }

                // Сохраняем сдвиги (biases)
                writer.Write(biases.Length);
                foreach (var layerBiases in biases)
                {
                    writer.Write(layerBiases.Length);
                    foreach (var bias in layerBiases)
                        writer.Write(bias);
                }
            }
        }
    }
}