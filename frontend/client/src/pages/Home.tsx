/**
 * Home — Main landing page with CV upload, AI analysis,
 * and live job matching via Adzuna + Reed APIs.
 */
import { useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import Navbar from "@/components/Navbar";
import StatsSection from "@/components/StatsSection";
import Footer from "@/components/Footer";
import PricingModal from "@/components/PricingModal";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Upload, FileText, CheckCircle2, Sparkles, ArrowRight,
  ExternalLink, MapPin, Building2, Loader2, AlertCircle,
  ChevronDown, ChevronUp, Shield,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { toast } from "sonner";

// ─── Constants ────────────────────────────────────────────────────────────────

const HERO_IMAGE = "https://d2xsxph8kpxj0f.cloudfront.net/310519663385886690/REzhPHfbXGKqcVNEDV6trD/hero-illustration-bW8LYQfFteFPUUhcHzUeVT.webp";

// ─── Types ────────────────────────────────────────────────────────────────────

interface JobMatch {
  id: string;
  jobTitle: string;
  company: string;
  location: string;
  salary: string;
  fitScore: number;
  fitReason: string;
  jobLink: string;
  jobDescription: string;
  source: string;
}

// ─── Step Indicator ───────────────────────────────────────────────────────────

type AnalysisStep = "idle" | "uploading" | "analysing" | "matching" | "done" | "error";

const STEP_LABELS: Record<AnalysisStep, string> = {
  idle: "Ready to upload",
  uploading: "Uploading your CV...",
  analysing: "AI is analysing your CV & scraping live jobs...",
  matching: "Matching you with sponsor-verified employers...",
  done: "Analysis complete!",
  error: "Something went wrong",
};

const STEP_PROGRESS: Record<AnalysisStep, number> = {
  idle: 0,
  uploading: 15,
  analysing: 45,
  matching: 75,
  done: 100,
  error: 0,
};

// ─── Score Helpers ────────────────────────────────────────────────────────────

function getScoreColor(score: number) {
  if (score >= 80) return { bg: "bg-emerald-100 dark:bg-emerald-900/30", text: "text-emerald-700 dark:text-emerald-400", ring: "ring-emerald-200 dark:ring-emerald-800" };
  if (score >= 60) return { bg: "bg-blue-100 dark:bg-blue-900/30", text: "text-blue-700 dark:text-blue-400", ring: "ring-blue-200 dark:ring-blue-800" };
  if (score >= 40) return { bg: "bg-amber-100 dark:bg-amber-900/30", text: "text-amber-700 dark:text-amber-400", ring: "ring-amber-200 dark:ring-amber-800" };
  return { bg: "bg-red-100 dark:bg-red-900/30", text: "text-red-700 dark:text-red-400", ring: "ring-red-200 dark:ring-red-800" };
}

function getScoreLabel(score: number) {
  if (score >= 80) return "Excellent Match";
  if (score >= 60) return "Good Match";
  if (score >= 40) return "Partial Match";
  return "Low Match";
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [step, setStep] = useState<AnalysisStep>("idle");
  const [matches, setMatches] = useState<JobMatch[]>([]);
  const [totalJobs, setTotalJobs] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [expandedJob, setExpandedJob] = useState<string | null>(null);
  const [showAllMatches, setShowAllMatches] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [pricingModalOpen, setPricingModalOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  // ─── File Handler ─────────────────────────────────────────────────────────

  const handleFile = useCallback(async (file: File) => {
    const validTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
    ];
    if (!validTypes.includes(file.type)) {
      toast.error("Please upload a PDF or DOCX file.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File too large. Maximum 10MB.");
      return;
    }

    setFileName(file.name);
    setStep("uploading");
    setErrorMsg(null);
    setMatches([]);

    try {
      const stepTimer1 = setTimeout(() => setStep("analysing"), 1500);
      const stepTimer2 = setTimeout(() => setStep("matching"), 8000);

      const result = await api.matchJobs(file);

      clearTimeout(stepTimer1);
      clearTimeout(stepTimer2);

      const mapped: JobMatch[] = (result.recommendations || []).map((rec: Record<string, any>, i: number) => ({
        id: String(i),
        jobTitle: rec["Job Title"] || "",
        company: rec["Company"] || "",
        location: rec["Location"] || "",
        salary: rec["Salary Min"] != null && rec["Salary Max"] != null
          ? `£${(rec["Salary Min"] / 1000).toFixed(0)}k – £${(rec["Salary Max"] / 1000).toFixed(0)}k`
          : "",
        fitScore: rec["Relevance Score"] || 0,
        fitReason: rec["Why"] || "",
        jobLink: rec["Job Link"] || "",
        jobDescription: rec["Job Description"] || "",
        source: rec["Source"] || "",
      }));

      setMatches(mapped);
      setTotalJobs(result.total_matched_jobs || 0);
      setStep("done");

      toast.success(
        `Found ${mapped.length} matching roles from ${result.sources?.adzuna ?? 0} Adzuna + ${result.sources?.reed ?? 0} Reed jobs!`
      );

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 300);
    } catch (error: any) {
      setStep("error");
      setErrorMsg(error.message || "Analysis failed. Please try again.");
      toast.error(error.message || "Analysis failed");
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }, [handleFile]);

  return (
    <div className="min-h-screen bg-background">
      <Navbar onOpenPricing={() => setPricingModalOpen(true)} />

      {/* ─── Hero Section ──────────────────────────────────────────────────── */}
      <section className="relative overflow-hidden">
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
                  Upload Your CV,{" "}
                  <span className="relative inline-block">
                    <span className="relative z-10 text-[oklch(0.488_0.243_264.376)]">Get Matched</span>
                    <span className="absolute bottom-1 left-0 right-0 h-3 bg-[oklch(0.488_0.243_264.376/0.12)] rounded-full -z-0" />
                  </span>{" "}
                  Instantly
                </h1>
              </motion.div>

              {/* Subheadline */}
              <motion.p
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.35 }}
                className="text-lg sm:text-xl text-muted-foreground leading-relaxed max-w-xl"
              >
                AI analyses your CV → Scrapes live jobs from Adzuna &amp; Reed → Matches you with UK Visa Sponsor employers → Ranked by relevance score.
              </motion.p>

              {/* Upload Area */}
              <motion.div
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.5 }}
              >
                <AnimatePresence mode="wait">
                  {step === "idle" || step === "error" ? (
                    <motion.div
                      key="upload"
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div
                        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                        onDragLeave={() => setIsDragging(false)}
                        onDrop={handleDrop}
                        className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all duration-300 ${
                          isDragging
                            ? "border-[oklch(0.488_0.243_264.376)] bg-[oklch(0.488_0.243_264.376/0.05)] scale-[1.02]"
                            : "border-border hover:border-[oklch(0.488_0.243_264.376/0.4)] hover:bg-[oklch(0.488_0.243_264.376/0.02)]"
                        }`}
                      >
                        <div className="space-y-4">
                          <div className="w-14 h-14 mx-auto rounded-2xl bg-[oklch(0.488_0.243_264.376/0.08)] flex items-center justify-center">
                            <Upload className="w-7 h-7 text-[oklch(0.488_0.243_264.376)]" />
                          </div>
                          <div>
                            <p className="text-base font-semibold text-foreground mb-1">
                              Drop your CV here or click to upload
                            </p>
                            <p className="text-sm text-muted-foreground">PDF or DOCX — max 10MB</p>
                          </div>
                          <Button
                            onClick={() => fileInputRef.current?.click()}
                            className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold shadow-lg shadow-[oklch(0.488_0.243_264.376/0.25)] hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.35)] transition-all duration-300 px-8 py-6 text-base"
                          >
                            <Upload className="w-5 h-5 mr-2" />
                            Upload CV
                          </Button>
                          <input
                            ref={fileInputRef}
                            type="file"
                            className="hidden"
                            accept=".pdf,.docx,.doc"
                            onChange={handleFileInput}
                          />
                        </div>

                        {step === "error" && errorMsg && (
                          <div className="mt-6 flex items-center gap-3 p-4 rounded-xl bg-destructive/10 border border-destructive/20">
                            <AlertCircle className="w-5 h-5 text-destructive flex-shrink-0" />
                            <div className="text-left">
                              <p className="text-sm font-medium text-destructive">{errorMsg}</p>
                              <p className="text-xs text-muted-foreground mt-1">Try uploading a different file or check the format.</p>
                            </div>
                          </div>
                        )}
                      </div>
                    </motion.div>
                  ) : step !== "done" ? (
                    <motion.div
                      key="processing"
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="border-2 border-[oklch(0.488_0.243_264.376/0.3)] rounded-2xl p-8 bg-[oklch(0.488_0.243_264.376/0.02)]">
                        <div className="text-center space-y-5">
                          <div className="w-14 h-14 mx-auto rounded-2xl bg-[oklch(0.488_0.243_264.376/0.08)] flex items-center justify-center">
                            <Loader2 className="w-7 h-7 text-[oklch(0.488_0.243_264.376)] animate-spin" />
                          </div>
                          <div>
                            <h3 className="font-display font-bold text-lg text-foreground mb-1">
                              {STEP_LABELS[step]}
                            </h3>
                            {fileName && (
                              <p className="text-sm text-muted-foreground flex items-center justify-center gap-2">
                                <FileText className="w-4 h-4" /> {fileName}
                              </p>
                            )}
                          </div>
                          <div className="max-w-sm mx-auto">
                            <Progress value={STEP_PROGRESS[step]} className="h-2" />
                            <div className="flex justify-between mt-2 text-[10px] text-muted-foreground">
                              <span>Upload</span>
                              <span>Analyse</span>
                              <span>Match</span>
                              <span>Done</span>
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            This usually takes 20-40 seconds...
                          </p>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <motion.div
                      key="done"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ duration: 0.5, type: "spring" }}
                    >
                      <div className="flex items-center gap-3 p-4 rounded-2xl bg-[oklch(0.696_0.17_162.48/0.08)] border border-[oklch(0.696_0.17_162.48/0.2)]">
                        <CheckCircle2 className="w-6 h-6 text-[oklch(0.596_0.145_163.225)] flex-shrink-0" />
                        <div className="flex-1">
                          <p className="text-sm font-semibold text-foreground">
                            CV analysed — {matches.length} job matches found
                          </p>
                          <p className="text-xs text-muted-foreground">
                            Scored from {totalJobs} sponsor-matched roles via Adzuna &amp; Reed
                          </p>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          className="rounded-xl text-xs"
                          onClick={() => { setStep("idle"); setMatches([]); }}
                        >
                          <Upload className="w-3.5 h-3.5 mr-1" />
                          New CV
                        </Button>
                      </div>

                      <button
                        onClick={() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
                        className="mt-3 text-sm text-[oklch(0.488_0.243_264.376)] hover:text-[oklch(0.441_0.243_264.376)] transition-colors flex items-center gap-1.5 mx-auto font-medium"
                      >
                        <ArrowRight className="w-3.5 h-3.5" />
                        Scroll down to see results
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>

              {/* Social proof */}
              {step === "idle" && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.7 }}
                  className="flex items-center gap-6 pt-2"
                >
                  <div className="flex -space-x-2">
                    {["bg-[oklch(0.488_0.243_264.376)]", "bg-[oklch(0.596_0.145_163.225)]", "bg-[oklch(0.7_0.15_60)]", "bg-[oklch(0.55_0.2_300)]"].map((bg, i) => (
                      <div key={i} className={`w-8 h-8 rounded-full ${bg} border-2 border-background flex items-center justify-center`}>
                        <span className="text-[10px] font-bold text-white">{["SC", "AK", "MR", "JL"][i]}</span>
                      </div>
                    ))}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    <span className="font-semibold text-foreground">2,400+</span> students matched this month
                  </p>
                </motion.div>
              )}
            </div>

            {/* Right: Illustration */}
            <motion.div
              initial={{ opacity: 0, x: 40 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.7, delay: 0.3 }}
              className="hidden lg:block relative"
            >
              <div className="relative">
                <div className="absolute -top-8 -right-8 w-32 h-32 bg-[oklch(0.488_0.243_264.376/0.08)] rounded-full blur-3xl" />
                <div className="absolute -bottom-8 -left-8 w-40 h-40 bg-[oklch(0.696_0.17_162.48/0.08)] rounded-full blur-3xl" />
                <img
                  src={HERO_IMAGE}
                  alt="International students collaborating on job applications with London skyline"
                  className="relative rounded-3xl shadow-2xl shadow-[oklch(0.488_0.243_264.376/0.1)] w-full"
                />
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

      {/* ─── Stats Section (only when idle) ────────────────────────────────── */}
      {step === "idle" && <StatsSection />}

      {/* ─── Results Section ───────────────────────────────────────────────── */}
      {step === "done" && (
        <section ref={resultsRef} className="py-12 sm:py-16">
          <div className="container max-w-5xl mx-auto px-4 space-y-10">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="flex items-center justify-between mb-5">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-[oklch(0.596_0.145_163.225/0.1)] flex items-center justify-center">
                    <Sparkles className="w-5 h-5 text-[oklch(0.596_0.145_163.225)]" />
                  </div>
                  <div>
                    <h2 className="font-display font-bold text-xl text-foreground">
                      {matches.length} Job Matches
                    </h2>
                    <p className="text-sm text-muted-foreground">
                      Scored from {totalJobs} sponsor-matched roles via Adzuna &amp; Reed
                    </p>
                  </div>
                </div>
              </div>

              {matches.length === 0 ? (
                <Card className="border-border">
                  <CardContent className="p-8 text-center">
                    <AlertCircle className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                    <p className="text-foreground font-medium">No matching jobs found</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      Try uploading a different CV or try again later when new jobs are posted.
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="space-y-4">
                  {(showAllMatches ? matches : matches.slice(0, 10)).map((match, i) => {
                    const scoreColor = getScoreColor(match.fitScore);
                    const isExpanded = expandedJob === match.id;

                    return (
                      <motion.div
                        key={match.id}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05 }}
                      >
                        <Card className="border-border hover:shadow-md transition-all duration-200">
                          <CardContent className="p-5">
                            <div className="flex items-start gap-4">
                              <div className={`flex-shrink-0 w-14 h-14 rounded-xl ${scoreColor.bg} ring-1 ${scoreColor.ring} flex flex-col items-center justify-center`}>
                                <span className={`font-display font-bold text-lg ${scoreColor.text}`}>
                                  {match.fitScore}
                                </span>
                                <span className={`text-[9px] font-medium ${scoreColor.text}`}>
                                  FIT
                                </span>
                              </div>

                              <div className="flex-1 min-w-0">
                                <div className="flex items-start justify-between gap-3">
                                  <div>
                                    <h3 className="font-display font-semibold text-foreground text-base leading-tight">
                                      {match.jobTitle}
                                    </h3>
                                    <div className="flex items-center gap-3 mt-1.5 text-sm text-muted-foreground">
                                      <span className="flex items-center gap-1">
                                        <Building2 className="w-3.5 h-3.5" />
                                        {match.company}
                                      </span>
                                      {match.location && (
                                        <span className="flex items-center gap-1">
                                          <MapPin className="w-3.5 h-3.5" />
                                          {match.location}
                                        </span>
                                      )}
                                      {match.source && (
                                        <Badge variant="secondary" className="text-[10px] px-1.5 py-0 rounded">
                                          {match.source}
                                        </Badge>
                                      )}
                                    </div>
                                  </div>
                                  <Badge className={`${scoreColor.bg} ${scoreColor.text} border-0 text-xs whitespace-nowrap`}>
                                    {getScoreLabel(match.fitScore)}
                                  </Badge>
                                </div>

                                <p className="text-sm text-muted-foreground mt-2 leading-relaxed">
                                  {match.fitReason}
                                </p>

                                {match.salary && match.salary !== " - " && (
                                  <p className="text-xs text-muted-foreground mt-1.5">
                                    Salary: {match.salary}
                                  </p>
                                )}

                                <div className="flex items-center gap-2 mt-3">
                                  {match.jobLink && (
                                    <Button
                                      variant="outline"
                                      size="sm"
                                      className="rounded-lg text-xs h-8"
                                      onClick={() => window.open(match.jobLink, "_blank")}
                                    >
                                      <ExternalLink className="w-3.5 h-3.5 mr-1" />
                                      View Job
                                    </Button>
                                  )}
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    className="rounded-lg text-xs h-8"
                                    onClick={() => setExpandedJob(isExpanded ? null : match.id)}
                                  >
                                    {isExpanded ? (
                                      <><ChevronUp className="w-3.5 h-3.5 mr-1" /> Less</>
                                    ) : (
                                      <><ChevronDown className="w-3.5 h-3.5 mr-1" /> Details</>
                                    )}
                                  </Button>
                                </div>

                                <AnimatePresence>
                                  {isExpanded && match.jobDescription && (
                                    <motion.div
                                      initial={{ opacity: 0, height: 0 }}
                                      animate={{ opacity: 1, height: "auto" }}
                                      exit={{ opacity: 0, height: 0 }}
                                      className="overflow-hidden"
                                    >
                                      <div className="mt-3 p-4 bg-muted/50 rounded-xl text-sm text-foreground leading-relaxed max-h-60 overflow-y-auto">
                                        {match.jobDescription.substring(0, 1500)}
                                        {match.jobDescription.length > 1500 && "..."}
                                      </div>
                                    </motion.div>
                                  )}
                                </AnimatePresence>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      </motion.div>
                    );
                  })}

                  {matches.length > 10 && !showAllMatches && (
                    <div className="text-center pt-2">
                      <Button
                        variant="outline"
                        className="rounded-xl"
                        onClick={() => setShowAllMatches(true)}
                      >
                        Show All {matches.length} Matches
                        <ArrowRight className="w-4 h-4 ml-1.5" />
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </motion.div>
          </div>
        </section>
      )}

      <Footer />
      <PricingModal
        isOpen={pricingModalOpen}
        onClose={() => setPricingModalOpen(false)}
      />
    </div>
  );
}
