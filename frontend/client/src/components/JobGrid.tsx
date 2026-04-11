/**
 * JobGrid — 3-column responsive grid of job cards with search/filter
 * Design: Warm Modernism — staggered entrance, warm section header
 * Free users see only 3 matches; Custom Plan unlocks all.
 */
import { useState, useMemo } from "react";
import { Search, SlidersHorizontal, Briefcase, Lock, Crown } from "lucide-react";
import { motion } from "framer-motion";
import JobCard from "./JobCard";
import type { JobListing } from "@/lib/data";

const FREE_TIER_LIMIT = 3;

interface JobGridProps {
  jobs: JobListing[];
  onGenerateEmail: (job: JobListing) => void;
  onViewHiringManager: (job: JobListing) => void;
  isPaidUser?: boolean;
  onOpenPricing?: () => void;
}

export default function JobGrid({ jobs, onGenerateEmail, onViewHiringManager, isPaidUser = false, onOpenPricing }: JobGridProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterNewEntrant, setFilterNewEntrant] = useState(false);
  const [sortBy, setSortBy] = useState<"recent" | "salary">("recent");

  const filteredJobs = useMemo(() => {
    let result = [...jobs];

    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      result = result.filter(
        (j) =>
          j.company.toLowerCase().includes(term) ||
          j.jobTitle.toLowerCase().includes(term) ||
          j.location.toLowerCase().includes(term) ||
          j.industry.toLowerCase().includes(term)
      );
    }

    if (filterNewEntrant) {
      result = result.filter((j) => j.isNewEntrantFriendly);
    }

    if (sortBy === "salary") {
      result.sort((a, b) => b.salaryMax - a.salaryMax);
    } else {
      result.sort((a, b) => a.postedDays - b.postedDays);
    }

    return result;
  }, [jobs, searchTerm, filterNewEntrant, sortBy]);

  const visibleJobs = isPaidUser ? filteredJobs : filteredJobs.slice(0, FREE_TIER_LIMIT);
  const lockedJobs = isPaidUser ? [] : filteredJobs.slice(FREE_TIER_LIMIT);
  const hasLockedJobs = lockedJobs.length > 0;

  return (
    <section className="py-16 md:py-20">
      <div className="container">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-10"
        >
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center">
              <Briefcase className="w-5 h-5 text-[oklch(0.488_0.243_264.376)]" />
            </div>
            <h2 className="font-display font-bold text-2xl sm:text-3xl text-foreground">Your Matched Jobs</h2>
          </div>
          <p className="text-muted-foreground text-base max-w-xl">
            Companies with verified UK visa sponsorship licences and ≤100 employees, matched to your profile.
          </p>
        </motion.div>

        {/* Search & Filters */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.4, delay: 0.1 }}
          className="flex flex-col sm:flex-row gap-3 mb-8"
        >
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search by company, role, location, or industry..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-11 pr-4 py-3 rounded-xl border border-border bg-card text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-[oklch(0.488_0.243_264.376/0.3)] focus:border-[oklch(0.488_0.243_264.376/0.5)] transition-all duration-200"
            />
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setFilterNewEntrant(!filterNewEntrant)}
              className={`px-4 py-3 rounded-xl text-sm font-medium border transition-all duration-200 flex items-center gap-2 ${
                filterNewEntrant
                  ? "bg-[oklch(0.696_0.17_162.48/0.1)] border-[oklch(0.696_0.17_162.48/0.3)] text-[oklch(0.596_0.145_163.225)]"
                  : "bg-card border-border text-muted-foreground hover:border-[oklch(0.696_0.17_162.48/0.3)]"
              }`}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              New Entrant Only
            </button>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as "recent" | "salary")}
              className="px-4 py-3 rounded-xl text-sm font-medium border border-border bg-card text-foreground focus:outline-none focus:ring-2 focus:ring-[oklch(0.488_0.243_264.376/0.3)] transition-all duration-200"
            >
              <option value="recent">Most Recent</option>
              <option value="salary">Highest Salary</option>
            </select>
          </div>
        </motion.div>

        {/* Results count */}
        <div className="flex items-center justify-between mb-6">
          <p className="text-sm text-muted-foreground">
            Showing <span className="font-semibold text-foreground">{visibleJobs.length}</span> of {filteredJobs.length} matches
            {!isPaidUser && filteredJobs.length > FREE_TIER_LIMIT && (
              <span className="ml-2 text-[oklch(0.488_0.243_264.376)]">
                ({filteredJobs.length - FREE_TIER_LIMIT} more with Custom Plan)
              </span>
            )}
          </p>
          {!isPaidUser && filteredJobs.length > FREE_TIER_LIMIT && (
            <button
              onClick={onOpenPricing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[oklch(0.488_0.243_264.376)] text-white text-xs font-medium hover:bg-[oklch(0.428_0.243_264.376)] transition-colors"
            >
              <Crown className="w-3.5 h-3.5" />
              Unlock All
            </button>
          )}
        </div>

        {/* Visible Grid */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
          {visibleJobs.map((job, i) => (
            <JobCard
              key={job.id}
              job={job}
              index={i}
              onGenerateEmail={onGenerateEmail}
              onViewHiringManager={onViewHiringManager}
            />
          ))}
        </div>

        {/* Locked Jobs Overlay */}
        {hasLockedJobs && (
          <div className="relative mt-5">
            {/* Blurred preview of locked jobs */}
            <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5 blur-[6px] pointer-events-none select-none opacity-60">
              {lockedJobs.slice(0, 6).map((job, i) => (
                <JobCard
                  key={job.id}
                  job={job}
                  index={i + FREE_TIER_LIMIT}
                  onGenerateEmail={() => {}}
                  onViewHiringManager={() => {}}
                />
              ))}
            </div>

            {/* Paywall overlay */}
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.3 }}
                className="bg-card/95 backdrop-blur-sm border border-border rounded-2xl p-8 text-center max-w-md shadow-xl"
              >
                <div className="w-14 h-14 mx-auto rounded-2xl bg-[oklch(0.488_0.243_264.376/0.1)] flex items-center justify-center mb-4">
                  <Lock className="w-7 h-7 text-[oklch(0.488_0.243_264.376)]" />
                </div>
                <h3 className="font-display font-bold text-xl text-foreground mb-2">
                  {lockedJobs.length} More Matches Available
                </h3>
                <p className="text-muted-foreground text-sm mb-6 leading-relaxed">
                  Upgrade to Custom Plan to unlock all job matches, unlimited AI emails, and hiring manager contact details.
                </p>
                <button
                  onClick={onOpenPricing}
                  className="w-full py-3 px-6 rounded-xl bg-[oklch(0.488_0.243_264.376)] text-white font-semibold text-sm hover:bg-[oklch(0.428_0.243_264.376)] transition-all duration-200 flex items-center justify-center gap-2 shadow-lg shadow-[oklch(0.488_0.243_264.376/0.25)]"
                >
                  <Crown className="w-4 h-4" />
                  Upgrade to Custom Plan — from £29/mo
                </button>
              </motion.div>
            </div>
          </div>
        )}

        {filteredJobs.length === 0 && (
          <div className="text-center py-16">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-muted flex items-center justify-center mb-4">
              <Search className="w-7 h-7 text-muted-foreground" />
            </div>
            <p className="font-display font-semibold text-lg text-foreground mb-1">No matches found</p>
            <p className="text-sm text-muted-foreground">Try adjusting your search or filters</p>
          </div>
        )}
      </div>
    </section>
  );
}
