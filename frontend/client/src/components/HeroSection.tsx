/**
 * HeroSection — Asymmetric hero with headline, CV upload, and illustration
 * Design: Warm Modernism — 60/40 split, warm cream bg, bouncing match counter
 */
import { useState, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Upload, FileText, CheckCircle2, Sparkles, ArrowRight, Shield, Brain } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useLocation } from "wouter";

const HERO_IMAGE = "https://d2xsxph8kpxj0f.cloudfront.net/310519663385886690/REzhPHfbXGKqcVNEDV6trD/hero-illustration-bW8LYQfFteFPUUhcHzUeVT.webp";

interface HeroSectionProps {
  onUploadComplete: () => void;
  hasUploaded: boolean;
  matchCount: number;
  onScrollToMatches: () => void;
}

export default function HeroSection({ onUploadComplete, hasUploaded, matchCount, onScrollToMatches }: HeroSectionProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [showCounter, setShowCounter] = useState(false);
  const [animatedCount, setAnimatedCount] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [, navigate] = useLocation();

  const simulateUpload = () => {
    if (isUploading || hasUploaded) return;
    setIsUploading(true);
    setUploadProgress(0);

    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsUploading(false);
          onUploadComplete();
          setTimeout(() => setShowCounter(true), 300);
          return 100;
        }
        return prev + Math.random() * 15 + 5;
      });
    }, 150);
  };

  useEffect(() => {
    if (showCounter && animatedCount < matchCount) {
      const timer = setTimeout(() => {
        setAnimatedCount((prev) => Math.min(prev + 1, matchCount));
      }, 40);
      return () => clearTimeout(timer);
    }
  }, [showCounter, animatedCount, matchCount]);

  return (
    <section className="relative overflow-hidden">
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-[0.03]" style={{
        backgroundImage: `radial-gradient(oklch(0.488 0.243 264.376) 1px, transparent 1px)`,
        backgroundSize: '24px 24px'
      }} />

      <div className="container relative py-16 md:py-24 lg:py-28">
        <div className="grid lg:grid-cols-[1.15fr_0.85fr] gap-12 lg:gap-16 items-center">
          {/* Left: Content */}
          <div className="space-y-8">
            {/* Trust badge */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[oklch(0.696_0.17_162.48/0.1)] border border-[oklch(0.696_0.17_162.48/0.2)]"
            >
              <Shield className="w-4 h-4 text-[oklch(0.596_0.145_163.225)]" />
              <span className="text-sm font-medium text-[oklch(0.596_0.145_163.225)]">
                Verified GOV.UK Sponsor Data
              </span>
            </motion.div>

            {/* Headline */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h1 className="font-display font-extrabold text-4xl sm:text-5xl lg:text-[3.4rem] leading-[1.1] tracking-tight text-foreground">
                Find UK Visa-Sponsored Jobs at Companies{" "}
                <span className="relative inline-block">
                  <span className="relative z-10 text-[oklch(0.488_0.243_264.376)]">≤100 Employees</span>
                  <span className="absolute bottom-1 left-0 right-0 h-3 bg-[oklch(0.488_0.243_264.376/0.12)] rounded-full -z-0" />
                </span>
              </h1>
            </motion.div>

            {/* Subheadline */}
            <motion.p
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.35 }}
              className="text-lg sm:text-xl text-muted-foreground leading-relaxed max-w-xl"
            >
              Upload your CV once → Get matched with verified sponsors → Send personalised cold emails that get replies in days.
            </motion.p>

            {/* Upload Area */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 }}
            >
              <AnimatePresence mode="wait">
                {!hasUploaded ? (
                  <motion.div
                    key="upload"
                    exit={{ opacity: 0, scale: 0.95 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div
                      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                      onDragLeave={() => setIsDragging(false)}
                      onDrop={(e) => { e.preventDefault(); setIsDragging(false); simulateUpload(); }}
                      className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
                        isDragging
                          ? "border-[oklch(0.488_0.243_264.376)] bg-[oklch(0.488_0.243_264.376/0.05)] scale-[1.02]"
                          : isUploading
                          ? "border-[oklch(0.488_0.243_264.376/0.4)] bg-[oklch(0.488_0.243_264.376/0.03)]"
                          : "border-border hover:border-[oklch(0.488_0.243_264.376/0.4)] hover:bg-[oklch(0.488_0.243_264.376/0.02)]"
                      }`}
                    >
                      {isUploading ? (
                        <div className="space-y-4">
                          <div className="w-12 h-12 mx-auto rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
                            <FileText className="w-6 h-6 text-[oklch(0.488_0.243_264.376)] animate-pulse" />
                          </div>
                          <p className="text-sm font-medium text-foreground">Analysing your CV with AI...</p>
                          <div className="w-full max-w-xs mx-auto h-2 bg-muted rounded-full overflow-hidden">
                            <div
                              className="h-full bg-[oklch(0.488_0.243_264.376)] rounded-full transition-all duration-300 ease-out"
                              style={{ width: `${Math.min(uploadProgress, 100)}%` }}
                            />
                          </div>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          <div className="w-14 h-14 mx-auto rounded-2xl bg-[oklch(0.488_0.243_264.376/0.08)] flex items-center justify-center">
                            <Upload className="w-7 h-7 text-[oklch(0.488_0.243_264.376)]" />
                          </div>
                          <div>
                            <p className="text-base font-semibold text-foreground mb-1">
                              Drop your CV here or click to upload
                            </p>
                            <p className="text-sm text-muted-foreground">PDF, DOCX, or paste your LinkedIn URL</p>
                          </div>
                          <Button
                            onClick={() => simulateUpload()}
                            className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold shadow-lg shadow-[oklch(0.488_0.243_264.376/0.25)] hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.35)] transition-all duration-300 px-8 py-6 text-base"
                          >
                            <Upload className="w-5 h-5 mr-2" />
                            Upload CV
                          </Button>
                          <input ref={fileInputRef} type="file" className="hidden" accept=".pdf,.docx" onChange={simulateUpload} />
                        </div>
                      )}
                    </div>

                    {/* Real CV Analysis link */}
                    <button
                      onClick={() => navigate("/cv-analysis")}
                      className="mt-4 text-sm text-[oklch(0.488_0.243_264.376)] hover:text-[oklch(0.441_0.243_264.376)] transition-colors flex items-center gap-1.5 mx-auto font-medium"
                    >
                      <Brain className="w-3.5 h-3.5" />
                      Upload your real CV for AI-powered job matching
                    </button>

                    {/* Demo link */}
                    <button
                      onClick={simulateUpload}
                      className="mt-2 text-xs text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1.5 mx-auto"
                    >
                      <Sparkles className="w-3 h-3" />
                      Or try with demo CV (Sienna Chen)
                    </button>
                  </motion.div>
                ) : (
                  <motion.div
                    key="result"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, type: "spring" }}
                    className="space-y-5"
                  >
                    {/* Success state */}
                    <div className="flex items-center gap-3 p-4 rounded-2xl bg-[oklch(0.696_0.17_162.48/0.08)] border border-[oklch(0.696_0.17_162.48/0.2)]">
                      <CheckCircle2 className="w-6 h-6 text-[oklch(0.596_0.145_163.225)] flex-shrink-0" />
                      <div>
                        <p className="text-sm font-semibold text-foreground">CV uploaded — Sienna Chen</p>
                        <p className="text-xs text-muted-foreground">Parsons MS Strategic Design & Management</p>
                      </div>
                    </div>

                    {/* Match counter */}
                    {showCounter && (
                      <motion.div
                        initial={{ opacity: 0, scale: 0.3 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.6, type: "spring", bounce: 0.5 }}
                        className="flex items-center gap-4 p-5 rounded-2xl bg-card border border-border shadow-lg shadow-[oklch(0.488_0.243_264.376/0.08)]"
                      >
                        <div className="w-14 h-14 rounded-2xl bg-[oklch(0.488_0.243_264.376)] flex items-center justify-center shadow-md shadow-[oklch(0.488_0.243_264.376/0.3)]">
                          <span className="font-display font-extrabold text-2xl text-white">{animatedCount}</span>
                        </div>
                        <div>
                          <p className="font-display font-bold text-lg text-foreground">Matches Found</p>
                          <p className="text-sm text-muted-foreground">Verified sponsors with ≤100 employees</p>
                        </div>
                        <Button
                          onClick={onScrollToMatches}
                          className="ml-auto bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold shadow-md shadow-[oklch(0.488_0.243_264.376/0.25)] transition-all duration-300"
                        >
                          View All
                          <ArrowRight className="w-4 h-4 ml-1.5" />
                        </Button>
                      </motion.div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

            {/* Social proof */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.7 }}
              className="flex items-center gap-6 pt-2"
            >
              <div className="flex -space-x-2">
                {["bg-[oklch(0.488_0.243_264.376)]", "bg-[oklch(0.696_0.17_162.48)]", "bg-[oklch(0.828_0.189_84.429)]", "bg-[oklch(0.541_0.281_275.847)]"].map((bg, i) => (
                  <div key={i} className={`w-8 h-8 rounded-full ${bg} border-2 border-background flex items-center justify-center`}>
                    <span className="text-[10px] font-bold text-white">{["SC", "AK", "MR", "JL"][i]}</span>
                  </div>
                ))}
              </div>
              <p className="text-sm text-muted-foreground">
                <span className="font-semibold text-foreground">2,400+</span> students matched this month
              </p>
            </motion.div>
          </div>

          {/* Right: Illustration */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.7, delay: 0.3 }}
            className="hidden lg:block relative"
          >
            <div className="relative">
              {/* Decorative blobs */}
              <div className="absolute -top-8 -right-8 w-32 h-32 bg-[oklch(0.488_0.243_264.376/0.08)] rounded-full blur-3xl" />
              <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-[oklch(0.696_0.17_162.48/0.08)] rounded-full blur-3xl" />
              
              <img
                src={HERO_IMAGE}
                alt="International students collaborating on job applications with London skyline"
                className="relative rounded-3xl shadow-2xl shadow-[oklch(0.488_0.243_264.376/0.1)] w-full"
              />

              {/* Floating card */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 1 }}
                className="absolute -bottom-6 -left-6 bg-card rounded-2xl shadow-xl shadow-black/10 border border-border p-4 max-w-[200px]"
              >
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 rounded-full bg-[oklch(0.696_0.17_162.48)] animate-pulse" />
                  <span className="text-xs font-semibold text-[oklch(0.596_0.145_163.225)]">Live Matching</span>
                </div>
                <p className="text-xs text-muted-foreground">AI scanning 121,000+ verified sponsors in real-time</p>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
