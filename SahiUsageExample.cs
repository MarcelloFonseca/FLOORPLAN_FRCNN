using System.Drawing;

namespace FloorplanDetection
{
    /// <summary>
    /// Exemple d'intégration : SAHI + ONNX Runtime pour Faster R-CNN (torchvision export).
    /// </summary>
    public class OnnxFloorplanDetector : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly string[] _classNames;

        public OnnxFloorplanDetector(string onnxModelPath, string[] classNames)
        {
            var options = new SessionOptions();
            _session = new InferenceSession(onnxModelPath, options);
            _classNames = classNames;
        }

        public List<Detection> RunInference(Bitmap image)
        {
            int w = image.Width;
            int h = image.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

            var bmpData = image.LockBits(
                new Rectangle(0, 0, w, h),
                ImageLockMode.ReadOnly,
                PixelFormat.Format24bppRgb);

            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;
                int stride = bmpData.Stride;

                for (int y = 0; y < h; y++)
                {
                    byte* row = ptr + y * stride;
                    for (int x = 0; x < w; x++)
                    {
                        tensor[0, 2, y, x] = row[x * 3 + 0] / 255f; // B → canal 2
                        tensor[0, 1, y, x] = row[x * 3 + 1] / 255f; // G → canal 1
                        tensor[0, 0, y, x] = row[x * 3 + 2] / 255f; // R → canal 0
                    }
                }
            }

            image.UnlockBits(bmpData);

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("images", tensor)
            };

            using var results = _session.Run(inputs);

            var boxes = results.First(r => r.Name == "boxes").AsTensor<float>();
            var labels = results.First(r => r.Name == "labels").AsTensor<long>();
            var scores = results.First(r => r.Name == "scores").AsTensor<float>();

            int count = (int)scores.Length;
            var detections = new List<Detection>(count);

            for (int i = 0; i < count; i++)
            {
                int classId = (int)labels[i] - 1;

                detections.Add(new Detection
                {
                    X1 = boxes[i, 0],
                    Y1 = boxes[i, 1],
                    X2 = boxes[i, 2],
                    Y2 = boxes[i, 3],
                    Score = scores[i],
                    ClassId = classId,
                    ClassName = classId >= 0 && classId < _classNames.Length
                        ? _classNames[classId]
                        : $"class_{classId}"
                });
            }

            return detections;
        }

        public void Dispose() => _session?.Dispose();
    }

    public static class Program
    {
        public static void Main()
        {
            string onnxPath = "floorplan_door_only2.onnx";
            string imagePath = "TEST-PLAN4.png";
            string[] classNames = { "door" };

            using var onnx = new OnnxFloorplanDetector(onnxPath, classNames);

            var sahi = new SahiDetector(
                inferenceFunc: onnx.RunInference,
                classNames: classNames,
                sliceWidth: 768,
                sliceHeight: 768,
                overlapWidthRatio: 0.25f,
                overlapHeightRatio: 0.25f,
                confidenceThreshold: 0.50f,
                nmsThreshold: 0.5f,
                performStandardPred: true,
                classAgnostic: false
            );

            var detections = sahi.Predict(imagePath);

            Console.WriteLine($"Détections : {detections.Count}");
            foreach (var det in detections)
            {
                Console.WriteLine(
                    $"  {det.ClassName} ({det.Score:F2}) " +
                    $"[{det.X1:F0}, {det.Y1:F0}, {det.X2:F0}, {det.Y2:F0}]");
            }
        }
    }
}
