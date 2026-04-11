/**
 * Type definitions and client-side helpers for MicroSponsor AI.
 * Data is now fetched from the database via tRPC.
 */

export interface JobListing {
  id: number;
  company: string;
  location: string;
  employeeCount: number;
  jobTitle: string;
  salaryMin: number;
  salaryMax: number;
  isNewEntrantFriendly: boolean;
  isVerifiedSponsor: boolean;
  hiringManager: string;
  hiringManagerEmail: string;
  companyDescription: string;
  industry: string;
  postedDays: number;
}

export interface DemoCV {
  name: string;
  university: string;
  degree: string;
  skills: string[];
  experience: string;
  visaStatus: string;
}

export interface EmailTemplate {
  id: number;
  jobId: number;
  subject: string;
  body: string;
  company: string;
  hiringManager: string;
}

/**
 * Client-side email generation fallback when no pre-built template exists.
 */
export function generateEmail(job: JobListing, cv: DemoCV): EmailTemplate {
  const firstName = job.hiringManager.split(" ")[0];

  return {
    id: 0,
    jobId: job.id,
    subject: `${firstName}, ${job.company}: ${cv.skills[0].toLowerCase()} expertise for your ${job.industry.toLowerCase()} platform?`,
    company: job.company,
    hiringManager: job.hiringManager,
    body: `Hi ${firstName},

I came across ${job.company} and was genuinely impressed by what you're building — ${job.companyDescription.toLowerCase()}. The ${job.industry.toLowerCase()} space is evolving fast, and your approach stands out.

I'm ${cv.name}, completing my ${cv.degree} at ${cv.university}. Before grad school, I had ${cv.experience.toLowerCase()}. My skills in ${cv.skills.slice(0, 3).join(", ")} align closely with the ${job.jobTitle} role you're hiring for.

I have some specific ideas about how ${cv.skills[0].toLowerCase()} could enhance your product's user experience and help ${job.company} scale more effectively. I'd love to walk you through them.

I'm currently on a ${cv.visaStatus} — as a New Entrant, the reduced salary threshold makes sponsoring me genuinely low-risk for a company your size. I'd welcome the chance to discuss how I could contribute to your team.

Would you have 15 minutes this week for a quick call?

Best,
${cv.name}
Portfolio: ${cv.name.toLowerCase().replace(" ", "-")}.design
LinkedIn: linkedin.com/in/${cv.name.toLowerCase().replace(" ", "")}`,
  };
}
