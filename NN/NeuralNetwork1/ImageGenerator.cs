using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    /// <summary>
    /// Тип фигуры
    /// </summary>
    public enum Letter : byte { alpha = 0, beta, gamma, delta, epsilon, pi, omega, teta, psi, nu, undef };

    public class GenerateImage
    {
        /// <summary>
        /// Бинарное представление образа
        /// </summary>
        public bool[,] img = new bool[28, 28];

        //  private int margin = 5;
        private Random rand = new Random();

        /// <summary>
        /// Текущая сгенерированная фигура
        /// </summary>
        public Letter currentAuto = Letter.undef;

        /// <summary>
        /// Количество классов генерируемых фигур (4 - максимум)
        /// </summary>
        public int FigureCount { get; set; } = 4;

        /// <summary>
        /// Диапазон смещения центра фигуры (по умолчанию +/- 5 пикселов от центра)
        /// </summary>
        public int FigureCenterGitter { get; set; } = 5;

        /// <summary>
        /// Диапазон разброса размера фигур
        /// </summary>
        public int FigureSizeGitter { get; set; } = 5;

        /// <summary>
        /// Диапазон разброса размера фигур
        /// </summary>
        public int FigureSize { get; set; } = 20;

        /// <summary>
        /// Очистка образа
        /// </summary>
        public void ClearImage()
        {
            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    img[i, j] = false;
        }


        private Point GetLeftUpperPoint()
        {
            int x = 14 - FigureSize / 2 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int y = 14 - FigureSize / 2 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(x, y);
        }

        private Point GetRightDownPoint()
        {
            int x = 14 + FigureSize / 2 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int y = 14 + FigureSize / 2 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(x, y);
        }

        private Point GetCenterPoint()
        {
            int x = 14 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            int y = 14 + rand.Next(-FigureSizeGitter / 2, FigureSizeGitter / 2);
            return new Point(x, y);
        }



        private void Bresenham(int x, int y, int x2, int y2)
        {
            int w = x2 - x;
            int h = y2 - y;
            int dx1 = 0, dy1 = 0, dx2 = 0, dy2 = 0;
            if (w < 0) dx1 = -1; else if (w > 0) dx1 = 1;
            if (h < 0) dy1 = -1; else if (h > 0) dy1 = 1;
            if (w < 0) dx2 = -1; else if (w > 0) dx2 = 1;
            int longest = Math.Abs(w);
            int shortest = Math.Abs(h);

            if (!(longest > shortest))
            {
                longest = Math.Abs(h);
                shortest = Math.Abs(w);
                if (h < 0) dy2 = -1; else if (h > 0) dy2 = 1;
                dx2 = 0;
            }

            int numerator = longest >> 1;
            for (int i = 0; i <= longest; i++)
            {
                if (x >= 0 && x < 28 && y >= 0 && y < 28)
                    img[x, y] = true;
                numerator += shortest;
                if (!(numerator < longest))
                {
                    numerator -= longest;
                    x += dx1;
                    y += dy1;
                }
                else
                {
                    x += dx2;
                    y += dy2;
                }
            }
        }


        /// <summary>
        /// Возвращает битовое изображение для вывода образа
        /// </summary>
        /// <returns></returns>
        public Bitmap GenBitmap()
        {
            Bitmap drawArea = new Bitmap(28, 28);
            for (int i = 0; i < 28; ++i)
                for (int j = 0; j < 28; ++j)
                    if (img[i, j])
                        drawArea.SetPixel(i, j, Color.Black);
            return drawArea;
        }
    }

}