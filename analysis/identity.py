"""One visual identity per entity, shared by every figure in the paper (briefs/00-test-the-pairing.md).

Colour means the agent and nothing else; marker shape means the model and nothing else. A figure that needs another
meaning uses a neutral, never one of these hues. Agents without a colour (the three that appear only in Study 1) are
drawn in NEUTRAL.

The three agent hues plus DECILE_RED pass the dataviz validator all-pairs in light mode (worst normal-vision dE 18.8,
DECILE_RED against Hermes). OpenCode's amber is under 3:1 contrast on white, so every figure names the agents in text.
Orange #eb6834 and violet #4a3aa7 keep their meaning from brief 03 (runs that broke a task rule) and are not reused.
"""

AGENT_COLOUR = {"pi": "#2a78d6", "hermes": "#d55181", "opencode": "#c98500"}
NEUTRAL = "#8C96A3"          # agents with no colour of their own, and generic marks
DECILE_RED = "#9c1c1c"       # the 10th and 90th percentile marks (figure 1)

# (marker, hollow): GLM-5.3 Flash is hollow because it is the starting model that Studies 2 and 3 share
MODEL_MARKER = {"glm-5.3-flash": ("o", True), "deepseek-4.1-flash": ("D", False), "glm-5.3": ("s", False)}
MODEL_NAME = {"glm-5.3-flash": "GLM-5.3 Flash", "deepseek-4.1-flash": "DeepSeek 4.1 Flash", "glm-5.3": "GLM-5.3"}
AGENT_NAME = {"pi": "pi", "opencode": "OpenCode", "hermes": "Hermes", "claude": "Claude Code", "codex": "Codex",
              "openclaw": "OpenClaw"}


def agent_colour(harness):
    return AGENT_COLOUR.get(harness, NEUTRAL)


def marker_kw(model, colour, size, alpha=1.0, edge=0.9):
    """scatter() keywords for a mark of this model in this colour: hollow models get a coloured ring on white."""
    marker, hollow = MODEL_MARKER.get(model, ("o", False))
    size = size * 0.8 if marker == "D" else size          # a diamond reads larger than a circle of the same area
    if hollow:
        return dict(marker=marker, s=size, facecolor="white", edgecolor=colour, linewidth=max(edge, 1.1), alpha=alpha)
    return dict(marker=marker, s=size, color=colour, edgecolor="white", linewidth=edge, alpha=alpha)
