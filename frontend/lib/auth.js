export const DEMO_USER = {
  name: "HydroCast Demo",
  email: "demo@hydrocast.ai",
  password: "demo123",
};

export const USERS_KEY = "hydrocast_users";
export const AUTH_KEY = "hydrocast_auth";

function browserOnly() {
  return typeof window !== "undefined";
}

export function getStoredUsers() {
  if (!browserOnly()) return [];
  try {
    const raw = window.localStorage.getItem(USERS_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveUsers(users) {
  if (!browserOnly()) return;
  window.localStorage.setItem(USERS_KEY, JSON.stringify(users));
}

export function createUser({ name, email, password }) {
  const normalizedEmail = email.trim().toLowerCase();
  const users = getStoredUsers();
  const emailTaken = users.some((user) => user.email.toLowerCase() === normalizedEmail) || normalizedEmail === DEMO_USER.email;

  if (emailTaken) {
    return { ok: false, message: "An account with this email already exists." };
  }

  if ((password ?? "").length < 6) {
    return { ok: false, message: "Password must be at least 6 characters." };
  }

  const nextUsers = [...users, { name: name.trim(), email: normalizedEmail, password }];
  saveUsers(nextUsers);
  return { ok: true, user: { name: name.trim(), email: normalizedEmail } };
}

export function authenticateUser({ email, password }) {
  const normalizedEmail = email.trim().toLowerCase();
  if (normalizedEmail === DEMO_USER.email && password === DEMO_USER.password) {
    return {
      ok: true,
      user: {
        name: DEMO_USER.name,
        email: DEMO_USER.email,
        isAuthenticated: true,
      },
    };
  }

  const matchedUser = getStoredUsers().find(
    (user) => user.email.toLowerCase() === normalizedEmail && user.password === password,
  );

  if (!matchedUser) {
    return { ok: false, message: "Invalid credentials" };
  }

  return {
    ok: true,
    user: {
      name: matchedUser.name,
      email: matchedUser.email,
      isAuthenticated: true,
    },
  };
}

export function getStoredSession() {
  if (!browserOnly()) return null;
  try {
    const raw = window.localStorage.getItem(AUTH_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function persistSession(session, remember = true) {
  if (!browserOnly()) return;
  if (!remember) {
    window.sessionStorage.setItem(AUTH_KEY, JSON.stringify(session));
    window.localStorage.removeItem(AUTH_KEY);
    return;
  }
  window.localStorage.setItem(AUTH_KEY, JSON.stringify(session));
  window.sessionStorage.removeItem(AUTH_KEY);
}

export function getActiveSession() {
  if (!browserOnly()) return null;
  try {
    const local = window.localStorage.getItem(AUTH_KEY);
    const session = local ?? window.sessionStorage.getItem(AUTH_KEY);
    return session ? JSON.parse(session) : null;
  } catch {
    return null;
  }
}

export function clearSession() {
  if (!browserOnly()) return;
  window.localStorage.removeItem(AUTH_KEY);
  window.sessionStorage.removeItem(AUTH_KEY);
}
