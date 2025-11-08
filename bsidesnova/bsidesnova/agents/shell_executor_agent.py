from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import subprocess

class ShellExecutorAgent:
    def __init__(
        self,
        ollama_client,
        users: Dict[str, Dict[str, str]],
        *,
        required_role: Optional[str] = None,   # e.g., "admin"; if None, no role required
        auth_required: bool = False,           # if True, must authenticate successfully before executing
        sso_required: bool = False,            # if True, sso_username must be present for execution
        sso_match_username: bool = True,       # if True, sso_username must match the LLM-identified username
        sso_admin_if_admin_required: bool = True,  # if True and required_role set, SSO identity must also have that role
    ):
        self.ollama = ollama_client
        self.users = users


        self.required_role = required_role
        self.auth_required = auth_required
        self.sso_required = sso_required
        self.sso_match_username = sso_match_username
        self.sso_admin_if_admin_required = sso_admin_if_admin_required

        self._handlers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
            "authenticate": self._step_authenticate,
            "get_role": self._step_get_role,
            "get_command": self._step_get_command,
        }

        self._planner_system = (
        "You are a strict planner. Output ONLY a JSON array (no prose). "
        "Use any subset of these steps, in any order, each at most once:\n"
        "- authenticate: {\"username\": \"<str|null>\", \"password\": \"<str|null>\"}\n"
        "- get_role:     {\"username\": \"<str|null>\"}\n"
        "- get_command:  {\"query\": \"<original user query>\", \"command\": \"<str|null>\"}\n"
        "- execute:      {\"command\": \"<str|null>\"}\n"
        "Shape examples:\n"
        "[{\"get_role\": {\"username\": \"alice\"}}, {\"execute\": {\"command\": \"echo hi\"}}]\n"
        "[{\"authenticate\": {\"username\": \"alice\", \"password\": \"pw\"}}, {\"execute\": {\"command\": \"echo hi\"}}]\n"
        "Rules:\n"
        "- Output strictly the JSON array. No comments/markdown.\n"
        "- Extract values from the user query if present; else null.\n"
        "- Use keys exactly as shown."
        "- You will always include auth and get role"
        "- Command is always a linux command line expression. If expression is not directly provided, give the expression"
        )

    def get_response(
        self,
        user_query: str,
        *,
        sso_username: Optional[str] = None,
        return_plan: bool = True,
        return_trace: bool = True,
        return_json: bool = True,
    ) -> str:
        plan = self._plan(user_query)

        state: Dict[str, Any] = {
            "username": None,
            "authenticated": None,
            "role": None,
            "command": None,
        }
        self.execute = True
        trace: List[Dict[str, Any]] = []

        for raw_step in plan:
            if not isinstance(raw_step, dict) or not raw_step:
                trace.append({"invalid_step": {"raw": raw_step}})
                continue

            step_name, args = next(iter(raw_step.items()))
            if not isinstance(args, dict):
                args = {}

            if step_name == "execute":
                exec_result = self._run_execute_step(args=args, state=state, sso_username=sso_username)
                trace.append({"execute": exec_result})
                continue

            handler = self._handlers.get(step_name)
            if handler is None:
                trace.append({"unknown_step": {"name": step_name, "args": args}})
                continue

            try:
                result = handler(args, state)
                trace.append({step_name: result})
            except:
                trace.append({"failed_step": {"name": step_name, "args": args}})
                self.execute = False
                continue

        out = {
            "plan": plan if return_plan else None,
            "trace": trace if return_trace else None,
        }
        return json.dumps(out, ensure_ascii=False, indent=2) if return_json else str(out)

    def _plan(self, user_query: str) -> List[Dict[str, Any]]:
        prompt = (
            f"USER QUERY:\n{user_query}\n\n"
            f"Produce the step plan now."
        )
        raw = self.ollama.generate(prompt=prompt, system=self._planner_system)
        plan = self._parse_json(raw, default=[])
        return self._normalize_plan(plan)

    def _step_authenticate(self, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        username = args.get("username")
        password = args.get("password")
        if username is None or password is None:
            print("Failed to Authenticate. No user or Password")
            self.execute = False
            return {"username": username, "authenticated": False}
        rec = self.users.get(str(username))
        ok = bool(isinstance(rec, dict) and rec.get("password") == str(password))
        state["authenticated"] = ok
        if self.auth_required and not ok:
            self.execute = False
        return {"username": username, "authenticated": ok}

    def _step_get_role(self, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        username = args.get("username") or state.get("username")
        state["username"] = username
        role = None
        rec = self.users.get(str(username)) if username is not None else None
        if isinstance(rec, dict):
            role = rec.get("role")
        if role != self.required_role:
            self.execute = False
        state["role"] = role
        return {"username": username, "role": role}

    def _step_get_command(self, args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        command = args.get("command")
        if command:
            state["command"] = command
        return {"query": args.get("query"), "command": command}

    def _run_execute_step(
        self,
        *,
        args: Dict[str, Any],
        state: Dict[str, Any],
        sso_username: Optional[str],
    ) -> Dict[str, Any]:
        command = args.get("command") or state.get("command")
        if self.sso_required and self.sso_match_username != state.get("username") and (self.sso_admin_if_admin_required and state.get("role") != "admin"):
            self.execute = False
            return {"command": command, "executed": False, "returncode": None, "stdout": None, "stderr": None}
        if command and self.execute:
            rc, out, err = self._execute(command)
            return {"command": command, "executed": True, "returncode": rc, "stdout": out, "stderr": err}
        return {"command": command, "executed": False, "returncode": None, "stdout": None, "stderr": None}


    def _execute(self, command: str) -> Tuple[int, str, str]:
        proc = subprocess.run(command, shell=True, capture_output=True, text=True)
        return proc.returncode, proc.stdout, proc.stderr

    @staticmethod
    def _normalize_plan(plan: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not isinstance(plan, list):
            return out
        last_key: Optional[str] = None
        for item in plan:
            if not isinstance(item, dict) or not item:
                continue
            key, val = next(iter(item.items()))
            if key == last_key:
                continue
            out.append({key: val if isinstance(val, dict) else {}})
            last_key = key
        return out

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        try:
            s = raw.strip()
            start = s.find("[") if s.startswith("[") else s.find("{")
            end = max(s.rfind("]"), s.rfind("}"))
            if start == -1 or end == -1 or end <= start:
                return default
            return json.loads(s[start : end + 1])
        except Exception:
            return default