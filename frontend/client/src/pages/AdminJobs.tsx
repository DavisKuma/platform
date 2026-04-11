/**
 * AdminJobs — Admin page for viewing scraped jobs
 * Uses the Python API /api/scrape endpoint to fetch live jobs.
 */
import { useAuth } from "@/_core/hooks/useAuth";
import { api } from "@/lib/api";
import { useState, useMemo, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Loader2, ArrowLeft, Shield, Building2, MapPin, Briefcase,
  RefreshCw, Search,
} from "lucide-react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { useLocation } from "wouter";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

export default function AdminJobs() {
  const { user } = useAuth();
  const [, navigate] = useLocation();
  const [searchTerm, setSearchTerm] = useState("");
  const [jobs, setJobs] = useState<Record<string, any>[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sources, setSources] = useState<{ adzuna: number; reed: number }>({ adzuna: 0, reed: 0 });

  const handleScrape = useCallback(async () => {
    setIsLoading(true);
    try {
      const result = await api.scrape();
      setJobs(result.jobs || []);
      setSources(result.sources || { adzuna: 0, reed: 0 });
      toast.success(`Scraped ${result.total_jobs} jobs (${result.sources.adzuna} Adzuna + ${result.sources.reed} Reed)`);
    } catch (error: any) {
      toast.error(error.message || "Failed to scrape jobs");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const filteredJobs = useMemo(() => {
    if (!searchTerm.trim()) return jobs;
    const term = searchTerm.toLowerCase();
    return jobs.filter(
      (job) =>
        (job["Company"] || "").toLowerCase().includes(term) ||
        (job["Job Title"] || "").toLowerCase().includes(term) ||
        (job["Location"] || "").toLowerCase().includes(term)
    );
  }, [jobs, searchTerm]);

  return (
    <div className="min-h-screen bg-background">
      <Navbar activeSection="dashboard" onNavigate={() => {}} onOpenPricing={() => {}} />

      <div className="container max-w-6xl mx-auto px-4 pt-28 pb-16">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-3 mb-2">
            <Button variant="ghost" size="sm" className="rounded-xl" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Dashboard
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <h1 className="font-display font-bold text-3xl text-foreground mb-1 flex items-center gap-2">
                <Shield className="w-7 h-7 text-[oklch(0.488_0.243_264.376)]" />
                Admin: Live Job Scraper
              </h1>
              <p className="text-muted-foreground">
                {jobs.length} jobs loaded · {sources.adzuna} Adzuna + {sources.reed} Reed
              </p>
            </div>
            <Button
              onClick={handleScrape}
              disabled={isLoading}
              className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl font-display font-semibold"
            >
              {isLoading ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Scraping...</>
              ) : (
                <><RefreshCw className="w-4 h-4 mr-2" /> Scrape Jobs</>
              )}
            </Button>
          </div>

          {jobs.length > 0 && (
            <div className="relative mb-6">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search jobs by company, title, location..."
                className="rounded-xl pl-10"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
          )}
        </motion.div>

        {isLoading ? (
          <div className="flex justify-center py-16">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-16">
            <Building2 className="w-12 h-12 text-muted-foreground mx-auto mb-3 opacity-40" />
            <p className="text-muted-foreground">Click "Scrape Jobs" to fetch live jobs from Adzuna &amp; Reed APIs.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredJobs.slice(0, 100).map((job, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.01 }}>
                <Card className="border-border bg-card">
                  <CardContent className="p-4">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Building2 className="w-4 h-4 text-[oklch(0.488_0.243_264.376)] flex-shrink-0" />
                          <h3 className="font-display font-bold text-foreground truncate">{job["Company"] || "Unknown"}</h3>
                          {job["Source"] && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 font-medium">
                              {job["Source"]}
                            </span>
                          )}
                          {job["Match Type"] && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400 font-medium">
                              {job["Match Type"]}
                            </span>
                          )}
                        </div>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1"><Briefcase className="w-3 h-3" /> {job["Job Title"] || ""}</span>
                          <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> {job["Location"] || ""}</span>
                          {job["Salary Min"] != null && <span>£{(job["Salary Min"] / 1000).toFixed(0)}k–£{((job["Salary Max"] || 0) / 1000).toFixed(0)}k</span>}
                        </div>
                      </div>
                      {job["Job Link"] && (
                        <Button size="sm" variant="outline" className="rounded-xl text-xs" onClick={() => window.open(job["Job Link"], "_blank")}>
                          View Job
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
            {filteredJobs.length > 100 && (
              <p className="text-center text-sm text-muted-foreground py-4">Showing 100 of {filteredJobs.length} jobs</p>
            )}
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
}
