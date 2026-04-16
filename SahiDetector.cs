using System.Drawing;

namespace FloorplanDetection
{
    public struct Detection
    {
        public float X1, Y1, X2, Y2;
        public float Score;
        public int ClassId;
        public string ClassName;

        public float Area => Math.Max(0, X2 - X1) * Math.Max(0, Y2 - Y1);
    }

    public struct SliceBBox
    {
        public int X, Y, Width, Height;
    }

    public static class SlicingHelper
    {
        public static List<SliceBBox> GetSliceBBoxes(
            int imageWidth,
            int imageHeight,
            int sliceWidth = 768,
            int sliceHeight = 768,
            float overlapWidthRatio = 0.25f,
            float overlapHeightRatio = 0.25f)
        {
            var slices = new List<SliceBBox>();

            int overlapW = (int)(sliceWidth * overlapWidthRatio);
            int overlapH = (int)(sliceHeight * overlapHeightRatio);
            int stepX = sliceWidth - overlapW;
            int stepY = sliceHeight - overlapH;

            for (int y = 0; y < imageHeight; y += stepY)
            {
                for (int x = 0; x < imageWidth; x += stepX)
                {
                    int x2 = Math.Min(x + sliceWidth, imageWidth);
                    int y2 = Math.Min(y + sliceHeight, imageHeight);
                    int x1 = Math.Max(0, x2 - sliceWidth);
                    int y1 = Math.Max(0, y2 - sliceHeight);

                    slices.Add(new SliceBBox
                    {
                        X = x1,
                        Y = y1,
                        Width = x2 - x1,
                        Height = y2 - y1
                    });
                }
            }

            return slices
                .GroupBy(s => (s.X, s.Y, s.Width, s.Height))
                .Select(g => g.First())
                .ToList();
        }

        public static Bitmap CropTile(Bitmap source, SliceBBox slice)
        {
            var rect = new Rectangle(slice.X, slice.Y, slice.Width, slice.Height);
            return source.Clone(rect, source.PixelFormat);
        }
    }

    public static class NmsHelper
    {
        public static List<Detection> ApplyNms(
            List<Detection> detections,
            float iouThreshold = 0.5f,
            bool classAgnostic = false)
        {
            if (detections.Count == 0) return detections;

            var sorted = detections.OrderByDescending(d => d.Score).ToList();
            var keep = new List<Detection>();
            var suppressed = new bool[sorted.Count];

            for (int i = 0; i < sorted.Count; i++)
            {
                if (suppressed[i]) continue;
                keep.Add(sorted[i]);

                for (int j = i + 1; j < sorted.Count; j++)
                {
                    if (suppressed[j]) continue;

                    if (!classAgnostic && sorted[i].ClassId != sorted[j].ClassId)
                        continue;

                    float iou = ComputeIoU(sorted[i], sorted[j]);
                    if (iou >= iouThreshold)
                        suppressed[j] = true;
                }
            }

            return keep;
        }

        private static float ComputeIoU(Detection a, Detection b)
        {
            float interX1 = Math.Max(a.X1, b.X1);
            float interY1 = Math.Max(a.Y1, b.Y1);
            float interX2 = Math.Min(a.X2, b.X2);
            float interY2 = Math.Min(a.Y2, b.Y2);

            float interArea = Math.Max(0, interX2 - interX1) * Math.Max(0, interY2 - interY1);
            float unionArea = a.Area + b.Area - interArea;

            return unionArea > 0 ? interArea / unionArea : 0f;
        }
    }

    public class SahiDetector
    {
        private readonly int _sliceWidth;
        private readonly int _sliceHeight;
        private readonly float _overlapW;
        private readonly float _overlapH;
        private readonly float _confidenceThreshold;
        private readonly float _nmsThreshold;
        private readonly bool _performStandardPred;
        private readonly bool _classAgnostic;
        private readonly string[] _classNames;
        private readonly Func<Bitmap, List<Detection>> _inferenceFunc;

        public SahiDetector(
            Func<Bitmap, List<Detection>> inferenceFunc,
            string[] classNames,
            int sliceWidth = 768,
            int sliceHeight = 768,
            float overlapWidthRatio = 0.25f,
            float overlapHeightRatio = 0.25f,
            float confidenceThreshold = 0.50f,
            float nmsThreshold = 0.5f,
            bool performStandardPred = true,
            bool classAgnostic = false)
        {
            _inferenceFunc = inferenceFunc;
            _classNames = classNames;
            _sliceWidth = sliceWidth;
            _sliceHeight = sliceHeight;
            _overlapW = overlapWidthRatio;
            _overlapH = overlapHeightRatio;
            _confidenceThreshold = confidenceThreshold;
            _nmsThreshold = nmsThreshold;
            _performStandardPred = performStandardPred;
            _classAgnostic = classAgnostic;
        }

        public List<Detection> Predict(string imagePath)
        {
            using var image = new Bitmap(imagePath);
            return Predict(image);
        }

        public List<Detection> Predict(Bitmap image)
        {
            var allDetections = new List<Detection>();

            var slices = SlicingHelper.GetSliceBBoxes(
                image.Width, image.Height,
                _sliceWidth, _sliceHeight,
                _overlapW, _overlapH);

            foreach (var slice in slices)
            {
                using var tile = SlicingHelper.CropTile(image, slice);
                var tileDetections = _inferenceFunc(tile);

                foreach (var det in tileDetections)
                {
                    if (det.Score < _confidenceThreshold) continue;

                    allDetections.Add(new Detection
                    {
                        X1 = det.X1 + slice.X,
                        Y1 = det.Y1 + slice.Y,
                        X2 = det.X2 + slice.X,
                        Y2 = det.Y2 + slice.Y,
                        Score = det.Score,
                        ClassId = det.ClassId,
                        ClassName = det.ClassId < _classNames.Length
                            ? _classNames[det.ClassId]
                            : $"class_{det.ClassId}"
                    });
                }
            }

            if (_performStandardPred)
            {
                var fullDetections = _inferenceFunc(image);
                foreach (var det in fullDetections)
                {
                    if (det.Score < _confidenceThreshold) continue;

                    allDetections.Add(new Detection
                    {
                        X1 = det.X1,
                        Y1 = det.Y1,
                        X2 = det.X2,
                        Y2 = det.Y2,
                        Score = det.Score,
                        ClassId = det.ClassId,
                        ClassName = det.ClassId < _classNames.Length
                            ? _classNames[det.ClassId]
                            : $"class_{det.ClassId}"
                    });
                }
            }

            return NmsHelper.ApplyNms(allDetections, _nmsThreshold, _classAgnostic);
        }
    }
}