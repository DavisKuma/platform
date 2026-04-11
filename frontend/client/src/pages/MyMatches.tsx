/**
 * MyMatches — Saved jobs page (placeholder — requires backend endpoints)
 */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Bookmark, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { useLocation } from "wouter";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import PricingModal from "@/components/PricingModal";

export default function MyMatches() {
  const [, navigate] = useLocation();
  const [pricingModalOpen, setPricingModalOpen] = useState(false);

  return (
    <div className="min-h-screen bg-background">
      <Navbar activeSection="matches" onNavigate={() => {}} onOpenPricing={() => setPricingModalOpen(true)} />

      <div className="container max-w-5xl mx-auto px-4 pt-28 pb-16">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex items-center gap-3 mb-2">
            <Button variant="ghost" size="sm" className="rounded-xl" onClick={() => navigate("/dashboard")}>
              <ArrowLeft className="w-4 h-4 mr-1" /> Dashboard
            </Button>
          </div>
          <h1 className="font-display font-bold text-3xl text-foreground mb-2">My Saved Matches</h1>
          <p className="text-muted-foreground mb-8">0 saved jobs</p>
        </motion.div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-20">
          <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-4">
            <Bookmark className="w-8 h-8 text-muted-foreground" />
          </div>
          <p className="font-display font-semibold text-lg text-foreground mb-2">No saved matches yet</p>
          <p className="text-muted-foreground text-sm mb-6">Upload your CV on the home page to get matched with jobs, then save them here</p>
          <Button className="bg-[oklch(0.488_0.243_264.376)] hover:bg-[oklch(0.441_0.243_264.376)] text-white rounded-xl" onClick={() => navigate("/")}>
            Browse Jobs
          </Button>
        </motion.div>
      </div>

      <Footer />
      <PricingModal isOpen={pricingModalOpen} onClose={() => setPricingModalOpen(false)} />
    </div>
  );
}
