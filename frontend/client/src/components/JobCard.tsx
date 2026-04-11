/**
 * JobCard — Individual job listing card with company info, badges, and action buttons
 * Design: Warm Modernism — warm gradient hover, emerald badges, card lift effect
 */
import { Button } from "@/components/ui/button";
import { MapPin, Users, Shield, Sparkles, Lock, Clock, Building2 } from "lucide-react";
import { motion } from "framer-motion";
import type { JobListing } from "@/lib/data";

interface JobCardProps {
  job: JobListing;
  index: number;
  onGenerateEmail: (job: JobListing) => void;
  onViewHiringManager: (job: JobListing) => void;
}

export default function JobCard({ job, index, onGenerateEmail, onViewHiringManager }: JobCardProps) {
  const formatSalary = (amount: number) => {
    return `£${(amount / 1000).toFixed(0)}k`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: index * 0.06, ease: [0.16, 1, 0.3, 1] }}
      className="group relative bg-card rounded-2xl border border-border hover:border-[oklch(0.488_0.243_264.376/0.3)] shadow-sm hover:shadow-xl hover:shadow-[oklch(0.488_0.243_264.376/0.06)] transition-all duration-300 hover:-translate-y-1 overflow-hidden"
    >
      {/* Top accent line */}
      <div className="absolute top-0 left-0 right-0 h-[3px] bg-gradient-to-r from-[oklch(0.488_0.243_264.376)] to-[oklch(0.696_0.17_162.48)] opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

      <div className="p-5 sm:p-6">
        {/* Header: Company + Posted */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-xl bg-[oklch(0.488_0.243_264.376/0.08)] flex items-center justify-center flex-shrink-0">
              <Building2 className="w-5 h-5 text-[oklch(0.488_0.243_264.376)]" />
            </div>
            <div>
              <h3 className="font-display font-bold text-base text-foreground leading-tight">{job.company}</h3>
              <p className="text-xs text-muted-foreground mt-0.5">{job.industry}</p>
            </div>
          </div>
          <span className="text-xs text-muted-foreground flex items-center gap-1 flex-shrink-0">
            <Clock className="w-3 h-3" />
            {job.postedDays}d ago
          </span>
        </div>

        {/* Job Title */}
        <h4 className="font-display font-semibold text-lg text-foreground mb-3">{job.jobTitle}</h4>

        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-4 text-sm text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5" />
            {job.location}
          </span>
          <span className="flex items-center gap-1.5">
            <Users className="w-3.5 h-3.5" />
            {job.employeeCount} employees
          </span>
        </div>

        {/* Salary */}
        <div className="mb-4">
          <span className="font-display font-bold text-lg text-foreground">
            {formatSalary(job.salaryMin)} – {formatSalary(job.salaryMax)}
          </span>
          <span className="text-sm text-muted-foreground ml-1.5">/year</span>
        </div>

        {/* Badges */}
        <div className="flex flex-wrap gap-2 mb-5">
          {job.isVerifiedSponsor && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[oklch(0.488_0.243_264.376/0.08)] text-[oklch(0.488_0.243_264.376)] text-xs font-semibold">
              <Shield className="w-3 h-3" />
              Verified Sponsor
            </span>
          )}
          {job.isNewEntrantFriendly && (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[oklch(0.696_0.17_162.48/0.1)] text-[oklch(0.596_0.145_163.225)] text-xs font-semibold">
              <Sparkles className="w-3 h-3" />
              New Entrant Friendly
            </span>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-2">
          <Button
            onClick={() => onViewHiringManager(job)}
            variant="outline"
            className="flex-1 rounded-xl text-sm font-medium border-border hover:border-[oklch(0.488_0.243_264.376/0.3)] hover:bg-[oklch(0.488_0.243_264.376/0.04)] transition-all duration-200"
          >
            <Lock className="w-3.5 h-3.5 mr-1.5 text-muted-foreground" />
            View Hiring Manager
          </Button>
          <Button
            onClick={() => onGenerateEmail(job)}
            className="flex-1 rounded-xl text-sm font-semibold bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white shadow-md shadow-[oklch(0.488_0.243_264.376/0.2)] hover:shadow-lg hover:shadow-[oklch(0.488_0.243_264.376/0.3)] transition-all duration-300"
          >
            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
            Generate Email
          </Button>
        </div>
      </div>
    </motion.div>
  );
}
