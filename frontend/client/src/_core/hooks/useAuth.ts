import { useCallback, useMemo } from "react";

/**
 * Simplified auth hook — no backend auth required.
 * All users are treated as authenticated so the CV upload flow works.
 * Wire this to Supabase Auth later if you need real user accounts.
 */
export function useAuth(_options?: { redirectOnUnauthenticated?: boolean; redirectPath?: string }) {
  const user = useMemo(() => ({ id: "local", name: "User", email: null, subscriptionTier: "free" }), []);
  const logout = useCallback(() => {}, []);

  return {
    user,
    loading: false,
    error: null,
    isAuthenticated: true,
    refresh: () => {},
    logout,
  };
}
