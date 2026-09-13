import { ChangeEvent, DragEvent, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { analyzeImageFromBackend } from "./services/api";


type AnalysisResult = {
  aiPercent: number;
  realPercent: number;
  verdict: string;
  confidence: number;
  reasons: string[];
  gradcam: string | null;
};


type HistoryItem = {
  id: number;
  fileName: string;
  previewUrl: string;
  result: AnalysisResult;
  timestamp: string;
};


function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(String(reader.result));

    reader.onerror = () =>
      reject(
        new Error("Unable to create image preview.")
      );

    reader.readAsDataURL(file);
  });
}


function wait(ms: number) {
  return new Promise<void>((resolve) => {
    window.setTimeout(resolve, ms);
  });
}


export default function App() {

  const [previewUrl, setPreviewUrl] =
    useState<string | null>(null);

  const [fileName, setFileName] =
    useState("No file selected");

  const [isDragging, setIsDragging] =
    useState(false);

  const [isAnalyzing, setIsAnalyzing] =
    useState(false);

  const [analysisProgress, setAnalysisProgress] =
    useState(0);

  const [error, setError] =
    useState<string | null>(null);

  const [result, setResult] =
    useState<AnalysisResult | null>(null);

  const [history, setHistory] =
    useState<HistoryItem[]>([]);

  const [showHistory, setShowHistory] =
    useState(false);


  // ==========================================
  // Clear current analysis
  // ==========================================

  const clearCurrentView = () => {

    setPreviewUrl(null);

    setFileName("No file selected");

    setResult(null);

    setError(null);

    setAnalysisProgress(0);
  };


  // ==========================================
  // Delete history item
  // ==========================================

  const deleteHistoryItem = (id: number) => {

    setHistory((prev) => {

      const removed = prev.find(
        (item) => item.id === id
      );

      const updated = prev.filter(
        (item) => item.id !== id
      );


      if (
        removed &&
        previewUrl === removed.previewUrl
      ) {
        clearCurrentView();
      }


      return updated;
    });
  };


  // ==========================================
  // Clear all history
  // ==========================================

  const clearAllHistory = () => {

    setHistory([]);

    clearCurrentView();
  };


  // ==========================================
  // Handle uploaded image
  // ==========================================

  const handleFile = async (
    file: File | null
  ) => {

    if (!file) {
      return;
    }


    // ------------------------------------------
    // Validate file type
    // ------------------------------------------

    if (!file.type.startsWith("image/")) {

      setError(
        "Please upload a valid image file."
      );

      return;
    }


    // ------------------------------------------
    // Validate file size
    // ------------------------------------------

    if (file.size > 10 * 1024 * 1024) {

      setError(
        "Maximum file size is 10MB."
      );

      return;
    }


    // ------------------------------------------
    // Reset UI
    // ------------------------------------------

    setError(null);

    setFileName(file.name);

    setResult(null);

    setIsAnalyzing(true);

    setAnalysisProgress(0);


    let progressTicker: number | null = null;


    try {

      const startTime =
        performance.now();


      // ----------------------------------------
      // Fake visual progress animation
      // ----------------------------------------

      progressTicker =
        window.setInterval(() => {

          setAnalysisProgress((prev) => {

            if (prev >= 90) {
              return prev;
            }


            return (
              prev +
              Math.max(
                1,
                Math.round(
                  (90 - prev) / 7
                )
              )
            );
          });

        }, 120);


      // ----------------------------------------
      // Create preview
      // ----------------------------------------

      const previewData =
        await fileToDataUrl(file);

      setPreviewUrl(previewData);


      // ----------------------------------------
      // Send image to FastAPI
      // ----------------------------------------

      const backendResult =
        await analyzeImageFromBackend(file);


      // ----------------------------------------
      // Create SignalScope result
      // ----------------------------------------

      const analysisResult: AnalysisResult = {

        aiPercent:
          backendResult.ai_percent,

        realPercent:
          backendResult.real_percent,

        verdict:
          backendResult.verdict,

        confidence:
          backendResult.confidence,

        gradcam:
          backendResult.gradcam,

        reasons: [

          `AI probability: ${backendResult.ai_percent}%`,

          `Real probability: ${backendResult.real_percent}%`,

          `Model confidence: ${backendResult.confidence}%`,

        ],
      };


      // ----------------------------------------
      // Keep animation visible for at least 1.8s
      // ----------------------------------------

      const elapsedTime =
        performance.now() - startTime;


      if (elapsedTime < 1800) {

        await wait(
          1800 - elapsedTime
        );
      }


      // ----------------------------------------
      // Stop progress animation
      // ----------------------------------------

      if (progressTicker) {

        window.clearInterval(
          progressTicker
        );

        progressTicker = null;
      }


      setAnalysisProgress(100);

      await wait(260);


      // ----------------------------------------
      // Show result
      // ----------------------------------------

      setResult(
        analysisResult
      );


      // ----------------------------------------
      // Add to history
      // ----------------------------------------

      setHistory((prev) => [

        {
          id: Date.now(),

          fileName: file.name,

          previewUrl: previewData,

          result: analysisResult,

          timestamp:
            new Date().toLocaleString(),
        },

        ...prev,

      ]);

    } catch (analysisError) {

      if (progressTicker) {

        window.clearInterval(
          progressTicker
        );

        progressTicker = null;
      }


      setError(

        analysisError instanceof Error

          ? analysisError.message

          : "Analysis failed. Please try a different image."

      );


      setAnalysisProgress(0);

    } finally {

      setIsAnalyzing(false);
    }
  };


  // ==========================================
  // File input
  // ==========================================

  const onInputChange = (
    event: ChangeEvent<HTMLInputElement>
  ) => {

    void handleFile(
      event.target.files?.[0] ?? null
    );
  };


  // ==========================================
  // Drag and drop
  // ==========================================

  const onDrop = (
    event: DragEvent<HTMLLabelElement>
  ) => {

    event.preventDefault();

    setIsDragging(false);

    void handleFile(
      event.dataTransfer.files?.[0] ?? null
    );
  };


  // ==========================================
  // UI
  // ==========================================

  return (

    <main
      className="
        relative
        min-h-screen
        overflow-hidden
        bg-[#071124]
        px-6
        py-10
        text-slate-100
      "
    >

      {/* Background */}

      <div
        className="
          pointer-events-none
          absolute
          inset-0
          bg-[radial-gradient(circle_at_top,rgba(37,99,235,0.18),transparent_45%),linear-gradient(180deg,rgba(10,18,37,0.45),rgba(7,17,36,0.94))]
        "
      />


      <div
        className="
          pointer-events-none
          absolute
          inset-0
          opacity-[0.08]
          [background-size:28px_28px]
          [background-image:linear-gradient(to_right,#a5b4fc_1px,transparent_1px),linear-gradient(to_bottom,#a5b4fc_1px,transparent_1px)]
        "
      />


      <section
        className="
          relative
          mx-auto
          flex
          w-full
          max-w-5xl
          flex-col
          items-center
          gap-9
        "
      >

        {/* =====================================
            Header
        ====================================== */}

        <motion.header

          initial={{
            opacity: 0,
            y: -16
          }}

          animate={{
            opacity: 1,
            y: 0
          }}

          transition={{
            duration: 0.55
          }}

          className="
            space-y-3
            pt-10
            text-center
          "
        >

          <p
            className="
              text-xs
              tracking-[0.28em]
              text-blue-200/80
            "
          >
            IMAGE AUTHENTICITY CHECK
          </p>


          <h1
            className="
              text-5xl
              font-semibold
              tracking-tight
              text-white
              sm:text-7xl
            "
          >
            Signal Scope
          </h1>


          <p
            className="
              mx-auto
              max-w-2xl
              text-sm
              text-slate-300/95
              sm:text-base
            "
          >
            Upload one image to estimate how likely it is
            AI-generated versus real photography.
          </p>

        </motion.header>


        {/* =====================================
            History Button
        ====================================== */}

        <div className="w-full">

          <button

            type="button"

            onClick={() =>
              setShowHistory(true)
            }

            className="
              ml-auto
              flex
              items-center
              gap-2
              rounded-md
              border
              border-slate-300/30
              bg-slate-900/55
              px-4
              py-2
              text-sm
              font-medium
              text-slate-100
              transition
              hover:border-blue-200/70
              hover:bg-slate-900
            "
          >

            <span>
              History
            </span>

            <span
              className="
                rounded-sm
                bg-slate-700
                px-2
                py-0.5
                text-xs
              "
            >
              {history.length}
            </span>

          </button>

        </div>


        {/* =====================================
            Upload Area
        ====================================== */}

        <motion.label

          initial={{
            opacity: 0,
            y: 24
          }}

          animate={{
            opacity: 1,
            y: 0
          }}

          transition={{
            duration: 0.6,
            delay: 0.12
          }}

          htmlFor="imageUpload"

          onDragOver={(event) => {

            event.preventDefault();

            setIsDragging(true);
          }}

          onDragLeave={() =>
            setIsDragging(false)
          }

          onDrop={onDrop}

          className={`

            group
            flex
            w-full
            cursor-pointer
            flex-col
            items-center
            justify-center
            gap-4
            rounded-2xl
            border
            border-dashed
            px-6
            py-12
            text-center
            transition

            ${
              isDragging

                ? "border-blue-300 bg-blue-400/10"

                : "border-slate-400/35 bg-slate-950/35 hover:border-blue-200/70 hover:bg-slate-950/55"
            }

          `}
        >

          <motion.div

            animate={{
              y: [0, -3, 0]
            }}

            transition={{
              duration: 2.4,
              repeat:
                Number.POSITIVE_INFINITY,
              ease: "easeInOut"
            }}

            className="
              rounded-full
              border
              border-slate-400/30
              bg-slate-900/70
              p-4
            "
          >

            <svg

              className="
                h-8
                w-8
                text-blue-200
              "

              viewBox="0 0 24 24"

              fill="none"

              stroke="currentColor"
            >

              <path
                d="M12 16V5"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              <path
                d="M7.75 9.25L12 5l4.25 4.25"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              <path
                d="M5 14.5v2.25A2.25 2.25 0 0 0 7.25 19h9.5A2.25 2.25 0 0 0 19 16.75V14.5"
                strokeWidth="1.6"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

            </svg>

          </motion.div>


          <div className="space-y-1">

            <p
              className="
                text-xl
                font-semibold
                text-white
              "
            >
              Upload an image
            </p>


            <p
              className="
                text-sm
                text-slate-300/95
              "
            >
              Drag and drop or click to choose
              (JPG, PNG, WEBP up to 10MB)
            </p>

          </div>


          <div
            className="
              rounded-md
              bg-blue-600
              px-6
              py-3
              text-sm
              font-medium
              text-white
              transition
              group-hover:bg-blue-500
            "
          >
            Choose Image
          </div>


          <input
            id="imageUpload"
            type="file"
            accept="image/*"
            className="hidden"
            onChange={onInputChange}
          />


          <p
            className="
              text-xs
              text-slate-400
            "
          >
            {fileName}
          </p>

        </motion.label>


        {/* Error */}

        {error && (

          <p
            className="
              text-sm
              text-rose-300
            "
          >
            {error}
          </p>

        )}


        {/* =====================================
            Analysis Section
        ====================================== */}

        <AnimatePresence mode="wait">

          {previewUrl && (

            <motion.section

              key={previewUrl}

              initial={{
                opacity: 0,
                y: 24
              }}

              animate={{
                opacity: 1,
                y: 0
              }}

              exit={{
                opacity: 0,
                y: -12
              }}

              transition={{
                duration: 0.5
              }}

              className="
                grid
                w-full
                gap-6
                pb-10
                md:grid-cols-2
              "
            >

              {/* =================================
                  Uploaded Image
              ================================== */}

              <div className="space-y-3">

                <p
                  className="
                    text-sm
                    font-medium
                    uppercase
                    tracking-[0.2em]
                    text-slate-300
                  "
                >
                  Uploaded image
                </p>


                <img
                  src={previewUrl}
                  alt="Uploaded preview"
                  className="
                    h-[320px]
                    w-full
                    rounded-xl
                    border
                    border-slate-300/30
                    object-cover
                  "
                />

              </div>


              {/* =================================
                  Result Card
              ================================== */}

              <div
                className="
                  space-y-4
                  rounded-xl
                  border
                  border-slate-300/30
                  bg-slate-950/45
                  p-6
                  backdrop-blur-sm
                "
              >

                <p
                  className="
                    text-sm
                    font-medium
                    uppercase
                    tracking-[0.2em]
                    text-slate-300
                  "
                >
                  Analysis result
                </p>


                {/* =================================
                    Loading
                ================================== */}

                {isAnalyzing && (

                  <div
                    className="
                      space-y-3
                      pt-4
                    "
                  >

                    <div
                      className="
                        flex
                        items-center
                        justify-between
                        text-sm
                        text-slate-200
                      "
                    >

                      <p
                        className="
                          font-medium
                          text-white
                        "
                      >
                        Analyzing image...
                      </p>


                      <p>
                        {analysisProgress}%
                      </p>

                    </div>


                    <div
                      className="
                        h-2
                        overflow-hidden
                        rounded-full
                        bg-slate-700/80
                      "
                    >

                      <motion.div

                        initial={{
                          width: 0
                        }}

                        animate={{
                          width:
                            `${analysisProgress}%`
                        }}

                        transition={{
                          duration: 0.25,
                          ease: "easeOut"
                        }}

                        className="
                          h-full
                          bg-blue-400
                        "
                      />

                    </div>


                    <p
                      className="
                        text-xs
                        text-slate-300
                      "
                    >
                      Processing visual patterns
                      and authenticity signatures.
                    </p>

                  </div>

                )}


                {/* =================================
                    Result
                ================================== */}

                {!isAnalyzing && result && (

                  <motion.div

                    initial={{
                      opacity: 0
                    }}

                    animate={{
                      opacity: 1
                    }}

                    className="
                      space-y-5
                    "
                  >

                    {/* Verdict */}

                    <div>

                      <p
                        className="
                          text-2xl
                          font-semibold
                          text-white
                        "
                      >
                        {result.verdict}
                      </p>

                    </div>


                    {/* =================================
                        Probability Circle
                    ================================== */}

                    <div
                      className="
                        grid
                        items-center
                        gap-5
                        sm:grid-cols-[180px_1fr]
                      "
                    >

                      <motion.div

                        initial={{
                          scale: 0.8,
                          opacity: 0
                        }}

                        animate={{
                          scale: 1,
                          opacity: 1
                        }}

                        transition={{
                          duration: 0.5
                        }}

                        className="
                          relative
                          mx-auto
                          h-40
                          w-40
                        "
                      >

                        <div
                          className="
                            h-full
                            w-full
                            rounded-full
                          "

                          style={{
                            background:
                              `conic-gradient(
                                #ef4444 0%
                                ${result.aiPercent}%,
                                #22c55e
                                ${result.aiPercent}%
                                100%
                              )`
                          }}
                        />


                        <div
                          className="
                            absolute
                            inset-5
                            flex
                            items-center
                            justify-center
                            rounded-full
                            bg-[#0b162d]
                            text-center
                          "
                        >

                          <div>

                            <p
                              className="
                                text-xs
                                uppercase
                                tracking-[0.2em]
                                text-slate-300
                              "
                            >
                              Result
                            </p>


                            <p
                              className="
                                text-xl
                                font-semibold
                                text-white
                              "
                            >
                              {result.aiPercent}% AI
                            </p>

                          </div>

                        </div>

                      </motion.div>


                      {/* Probabilities */}

                      <div
                        className="
                          space-y-3
                          text-sm
                        "
                      >

                        <div
                          className="
                            flex
                            items-center
                            justify-between
                            border-b
                            border-slate-300/20
                            pb-2
                          "
                        >

                          <div
                            className="
                              flex
                              items-center
                              gap-2
                              text-slate-200
                            "
                          >

                            <span
                              className="
                                h-2.5
                                w-2.5
                                rounded-full
                                bg-red-500
                              "
                            />

                            <span>
                              AI probability
                            </span>

                          </div>


                          <span
                            className="
                              font-semibold
                              text-white
                            "
                          >
                            {result.aiPercent}%
                          </span>

                        </div>


                        <div
                          className="
                            flex
                            items-center
                            justify-between
                          "
                        >

                          <div
                            className="
                              flex
                              items-center
                              gap-2
                              text-slate-200
                            "
                          >

                            <span
                              className="
                                h-2.5
                                w-2.5
                                rounded-full
                                bg-green-500
                              "
                            />

                            <span>
                              Real probability
                            </span>

                          </div>


                          <span
                            className="
                              font-semibold
                              text-white
                            "
                          >
                            {result.realPercent}%
                          </span>

                        </div>

                      </div>

                    </div>


                    {/* =================================
                        Explanation
                    ================================== */}

                    <ul
                      className="
                        space-y-2
                        border-t
                        border-slate-300/20
                        pt-4
                        text-sm
                        text-slate-300
                      "
                    >

                      {result.reasons.map(
                        (reason) => (

                          <li
                            key={reason}
                            className="
                              leading-relaxed
                            "
                          >
                            {reason}
                          </li>

                        )
                      )}

                    </ul>


                    {/* =================================
                        Confidence
                    ================================== */}

                    <div
                      className="
                        rounded-lg
                        border
                        border-blue-300/20
                        bg-blue-500/5
                        p-4
                      "
                    >

                      <div
                        className="
                          flex
                          items-center
                          justify-between
                        "
                      >

                        <span
                          className="
                            text-sm
                            text-slate-300
                          "
                        >
                          Model confidence
                        </span>


                        <span
                          className="
                            font-semibold
                            text-white
                          "
                        >
                          {result.confidence}%
                        </span>

                      </div>


                      <div
                        className="
                          mt-2
                          h-1.5
                          overflow-hidden
                          rounded-full
                          bg-slate-700
                        "
                      >

                        <motion.div

                          initial={{
                            width: 0
                          }}

                          animate={{
                            width:
                              `${result.confidence}%`
                          }}

                          transition={{
                            duration: 0.7
                          }}

                          className="
                            h-full
                            bg-blue-400
                          "
                        />

                      </div>

                    </div>


                    {/* =================================
                        Grad-CAM
                    ================================== */}

                    {result.gradcam && (

                      <motion.div

                        initial={{
                          opacity: 0,
                          y: 12
                        }}

                        animate={{
                          opacity: 1,
                          y: 0
                        }}

                        transition={{
                          duration: 0.45
                        }}

                        className="
                          space-y-3
                          border-t
                          border-slate-300/20
                          pt-5
                        "
                      >

                        <div>

                          <p
                            className="
                              text-sm
                              font-medium
                              uppercase
                              tracking-[0.2em]
                              text-slate-300
                            "
                          >
                            AI Attention Heatmap
                          </p>


                          <p
                            className="
                              mt-1
                              text-xs
                              leading-relaxed
                              text-slate-400
                            "
                          >
                            Highlighted regions show
                            areas that contributed to
                            the model's prediction.
                          </p>

                        </div>


                        <div
                          className="
                            overflow-hidden
                            rounded-xl
                            border
                            border-slate-300/20
                            bg-slate-900/50
                          "
                        >

                          <img
                            src={result.gradcam}
                            alt="Grad-CAM attention heatmap"
                            className="
                              h-auto
                              w-full
                              object-cover
                            "
                          />

                        </div>


                        <div
                          className="
                            flex
                            flex-wrap
                            items-center
                            gap-3
                            text-xs
                            text-slate-400
                          "
                        >

                          <span
                            className="
                              inline-block
                              h-3
                              w-3
                              rounded-full
                              bg-red-500
                            "
                          />

                          <span>
                            Higher model attention
                          </span>


                          <span
                            className="
                              ml-3
                              inline-block
                              h-3
                              w-3
                              rounded-full
                              bg-blue-500
                            "
                          />

                          <span>
                            Lower model attention
                          </span>

                        </div>

                      </motion.div>

                    )}

                  </motion.div>

                )}

              </div>

            </motion.section>

          )}

        </AnimatePresence>

      </section>


      {/* ========================================
          History Modal
      ========================================= */}

      <AnimatePresence>

        {showHistory && (

          <motion.div

            initial={{
              opacity: 0
            }}

            animate={{
              opacity: 1
            }}

            exit={{
              opacity: 0
            }}

            className="
              fixed
              inset-0
              z-40
              flex
              items-end
              justify-center
              bg-black/60
              p-4
              md:items-center
            "
          >

            <motion.div

              initial={{
                y: 24,
                opacity: 0
              }}

              animate={{
                y: 0,
                opacity: 1
              }}

              exit={{
                y: 24,
                opacity: 0
              }}

              transition={{
                duration: 0.25
              }}

              className="
                max-h-[80vh]
                w-full
                max-w-3xl
                overflow-hidden
                rounded-xl
                border
                border-slate-300/25
                bg-[#0b1529]
              "
            >

              {/* History Header */}

              <div
                className="
                  flex
                  items-center
                  justify-between
                  border-b
                  border-slate-300/20
                  px-5
                  py-4
                "
              >

                <h2
                  className="
                    text-lg
                    font-semibold
                    text-white
                  "
                >
                  Upload History
                </h2>


                <div
                  className="
                    flex
                    items-center
                    gap-2
                  "
                >

                  <button

                    type="button"

                    onClick={clearAllHistory}

                    className="
                      rounded-md
                      border
                      border-red-400/40
                      px-3
                      py-1.5
                      text-sm
                      text-red-200
                      transition
                      hover:bg-red-500/15
                    "
                  >
                    Clear History
                  </button>


                  <button

                    type="button"

                    onClick={() =>
                      setShowHistory(false)
                    }

                    className="
                      rounded-md
                      border
                      border-slate-300/30
                      px-3
                      py-1.5
                      text-sm
                      text-slate-200
                      transition
                      hover:bg-slate-800
                    "
                  >
                    Close
                  </button>

                </div>

              </div>


              {/* History Content */}

              <div
                className="
                  max-h-[65vh]
                  overflow-y-auto
                  px-5
                  py-4
                "
              >

                {history.length === 0 ? (

                  <p
                    className="
                      py-6
                      text-sm
                      text-slate-300
                    "
                  >
                    No images uploaded yet.
                  </p>

                ) : (

                  <div
                    className="
                      space-y-3
                    "
                  >

                    {history.map((item) => (

                      <div

                        key={item.id}

                        className="
                          flex
                          items-center
                          gap-3
                          rounded-lg
                          border
                          border-slate-300/20
                          bg-slate-900/45
                          p-3
                        "
                      >

                        {/* Open history item */}

                        <button

                          type="button"

                          onClick={() => {

                            setPreviewUrl(
                              item.previewUrl
                            );

                            setFileName(
                              item.fileName
                            );

                            setResult(
                              item.result
                            );

                            setShowHistory(false);

                          }}

                          className="
                            flex
                            min-w-0
                            flex-1
                            items-center
                            gap-3
                            text-left
                          "
                        >

                          <img
                            src={item.previewUrl}
                            alt={item.fileName}
                            className="
                              h-14
                              w-14
                              rounded-md
                              object-cover
                            "
                          />


                          <div
                            className="
                              min-w-0
                              flex-1
                            "
                          >

                            <p
                              className="
                                truncate
                                text-sm
                                font-medium
                                text-white
                              "
                            >
                              {item.fileName}
                            </p>


                            <p
                              className="
                                text-xs
                                text-slate-400
                              "
                            >
                              {item.timestamp}
                            </p>

                          </div>


                          <div
                            className="
                              text-right
                              text-xs
                            "
                          >

                            <p
                              className="
                                text-red-300
                              "
                            >
                              AI: {item.result.aiPercent}%
                            </p>


                            <p
                              className="
                                text-green-300
                              "
                            >
                              Real: {item.result.realPercent}%
                            </p>

                          </div>

                        </button>


                        {/* Delete */}

                        <button

                          type="button"

                          onClick={() =>
                            deleteHistoryItem(
                              item.id
                            )
                          }

                          className="
                            rounded-md
                            border
                            border-red-400/40
                            px-3
                            py-2
                            text-xs
                            text-red-200
                            transition
                            hover:bg-red-500/15
                          "
                        >
                          Delete
                        </button>

                      </div>

                    ))}

                  </div>

                )}

              </div>

            </motion.div>

          </motion.div>

        )}

      </AnimatePresence>

    </main>
  );
}