// Translates raw LangGraph SSE events into UI actions. Pure: no DOM, testable in Node.
//
// Wire format (stream_mode ["updates", "custom"], stream_subgraphs true):
//   metadata                      {run_id}
//   updates                       {<orchestrator node>: <update>} or {__interrupt__: [...]}
//   updates|conduct_research:<id> {<analyst node>: <update>}
//   custom|conduct_research:<id>  {type: "analyst_started" | "analyst_step", ...}
//   error                         {error, message}; error is the exception class name

export const SYNTHESIS_STEPS = ["write_report", "write_introduction", "write_conclusion", "finalize_report"];

export function messageText(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => (typeof part === "string" ? part : part?.type === "text" || part?.text ? part.text ?? "" : ""))
      .join("");
  }
  return "";
}

function usageOf(update) {
  if (!update || typeof update !== "object" || !("llm_calls" in update)) return null;
  return { llmCalls: update.llm_calls ?? 0, inputTokens: update.input_tokens ?? 0 };
}

function lastMessage(update) {
  const messages = update?.messages;
  return Array.isArray(messages) && messages.length ? messages[messages.length - 1] : null;
}

function analystActions(index, node, update) {
  const usage = usageOf(update);
  switch (node) {
    case "planner_node":
      return [{ kind: "plan", index, subQuestions: update?.sub_questions ?? [], usage }];
    case "researcher_node": {
      const message = lastMessage(update);
      return [
        {
          kind: "researcher_turn",
          index,
          turn: update?.researcher_turns ?? null,
          reasoning: messageText(message?.content).trim(),
          toolCalls: message?.tool_calls ?? [],
          usage,
        },
      ];
    }
    case "limit_tool_calls":
      return [{ kind: "tools_dispatched", index, calls: lastMessage(update)?.tool_calls ?? [] }];
    case "tool_node":
      return [
        {
          kind: "tool_results",
          index,
          results: (update?.messages ?? []).map((message) => ({
            id: message.tool_call_id,
            name: message.name,
            status: message.status ?? "success",
            content: messageText(message.content),
          })),
        },
      ];
    case "extract_findings":
      return [{ kind: "findings", index, findings: update?.research_findings ?? [], usage }];
    case "evaluate_research":
      return update
        ? [{ kind: "review", index, evaluation: update.evaluation ?? null, usage }]
        : [{ kind: "review_skipped", index }];
    case "writer_node":
      return [{ kind: "draft", index, draft: update?.draft ?? "", stopReason: update?.stop_reason ?? null, usage }];
    default:
      return [];
  }
}

function orchestratorActions(node, update) {
  switch (node) {
    case "__interrupt__": {
      const interrupt = Array.isArray(update) ? update[0] : update;
      return [{ kind: "feedback_requested", payload: interrupt?.value ?? {} }];
    }
    case "create_analysts":
      return [{ kind: "analysts_generated", analysts: update?.analysts ?? [], profile: update?.model_profile ?? null }];
    case "human_feedback":
      return [
        {
          kind: "feedback_resolved",
          feedback: update?.human_analyst_feedback ?? null,
          revisionCount: update?.revision_count ?? null,
        },
      ];
    case "conduct_research":
      return (update?.analyst_stats ?? []).map((stats) => ({ kind: "analyst_completed", stats }));
    case "finalize_report":
      return [{ kind: "final_report", report: update?.final_report ?? "", runStats: update?.run_stats ?? null }];
    default:
      return SYNTHESIS_STEPS.includes(node) ? [{ kind: "synthesis_step", step: node }] : [];
  }
}

export function createInterpreter() {
  const namespaces = new Map(); // "conduct_research:<task id>" -> analyst index

  function indexFor(namespace, hinted) {
    if (namespaces.has(namespace)) return namespaces.get(namespace);
    const index = Number.isInteger(hinted) ? hinted : namespaces.size;
    namespaces.set(namespace, index);
    return index;
  }

  return function interpret({ event, data }) {
    const [mode, namespace] = String(event).split("|");
    if (mode === "metadata") return data?.run_id ? [{ kind: "run_started", runId: data.run_id }] : [];
    if (mode === "error") {
      const message = data?.message || data?.error || (typeof data === "string" ? data : "Run failed");
      // The topic check rejects input before any analyst exists: a form error, not a failed run.
      if (data?.error === "UnresearchableTopicError") return [{ kind: "topic_rejected", message }];
      return [{ kind: "run_error", message }];
    }
    if (mode === "custom" && namespace && data && typeof data === "object") {
      const index = indexFor(namespace, data.analyst_index);
      if (data.type === "analyst_started") {
        return [
          { kind: "analyst_started", index, analyst: data.analyst, profile: data.profile, limits: data.limits },
        ];
      }
      if (data.type === "analyst_step") {
        return [
          {
            kind: "analyst_step",
            index,
            node: data.node,
            researchPass: data.research_pass ?? null,
            turn: data.turn ?? null,
            toolOutputs: data.tool_outputs ?? null,
            stopReason: data.stop_reason ?? null,
          },
        ];
      }
      return [];
    }
    if (mode === "updates" && data && typeof data === "object") {
      if (namespace) {
        const index = indexFor(namespace, null);
        return Object.entries(data).flatMap(([node, update]) => analystActions(index, node, update));
      }
      return Object.entries(data).flatMap(([node, update]) => orchestratorActions(node, update));
    }
    return [];
  };
}
