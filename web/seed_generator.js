const { app } = window.comfyAPI.app;

const MAX_SEED = 1125899906842624;

function randomSeed() {
  const values = new Uint32Array(2);
  crypto.getRandomValues(values);
  return Math.floor((values[0] * 0x100000000 + values[1]) / 16384);
}

app.registerExtension({
  name: "UtilitySuite.SeedGenerator",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const state = nodeData.input?.required?.state;
    if (nodeData.name !== "UtilitySuiteSeedGenerator" || !state?.[1]?.seed_generator) return;
    state[0] = "UTILITY_SEED_GENERATOR";

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const seed = Array.isArray(message?.seed) ? message.seed[0] : message?.seed;
      if (Number.isInteger(seed)) this._utilitySeedGeneratorExecuted?.(seed);
    };
  },
  getCustomWidgets() {
    return {
      UTILITY_SEED_GENERATOR(node, inputName, inputData) {
        let serialized = inputData[1]?.default || '{"seed":0,"randomize":true}';
        let state = { seed: 0, randomize: true };
        let lastSeed;
        let widget;

        const panel = document.createElement("div");
        panel.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:4px;";
        const seedInput = document.createElement("input");
        seedInput.type = "number";
        seedInput.min = "0";
        seedInput.max = String(MAX_SEED);
        seedInput.step = "1";
        seedInput.style.cssText = "box-sizing:border-box;width:100%;height:32px;padding:4px 8px;";

        const makeButton = (label, action) => {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = label;
          button.style.height = "34px";
          button.onclick = action;
          return button;
        };

        const sync = () => {
          serialized = JSON.stringify(state);
          widget.value = serialized;
          widget.callback?.(serialized);
          node.onWidgetChanged?.(inputName, serialized, undefined, widget);
          node.graph?.setDirtyCanvas(true, true);
          render();
        };

        const randomizeButton = makeButton("🎲 Randomize Each Time", () => {
          state.randomize = true;
          sync();
        });
        const fixedButton = makeButton("🎲 New Fixed Random", () => {
          state = { seed: randomSeed(), randomize: false };
          sync();
        });
        const lastButton = makeButton("♻️ (Use Last Queued Seed)", () => {
          if (lastSeed === undefined) return;
          state = { seed: lastSeed, randomize: false };
          sync();
        });
        seedInput.onchange = () => {
          const value = Math.trunc(Number(seedInput.value));
          state = { seed: Math.max(0, Math.min(MAX_SEED, value || 0)), randomize: false };
          sync();
        };
        panel.append(seedInput, randomizeButton, fixedButton, lastButton);

        const render = () => {
          seedInput.value = String(state.seed);
          randomizeButton.disabled = state.randomize;
          lastButton.disabled = lastSeed === undefined || (!state.randomize && state.seed === lastSeed);
          lastButton.textContent = lastSeed === undefined
            ? "♻️ (Use Last Queued Seed)"
            : `♻️ ${lastSeed}`;
        };
        const load = () => {
          try {
            const saved = JSON.parse(serialized);
            state = {
              seed: Number.isInteger(saved.seed) ? saved.seed : 0,
              randomize: saved.randomize !== false,
            };
          } catch (_) {
            state = { seed: 0, randomize: true };
          }
          render();
        };

        widget = node.addDOMWidget(inputName, "UTILITY_SEED_GENERATOR", panel, {
          getValue: () => serialized,
          setValue: (value) => {
            serialized = typeof value === "string" ? value : inputData[1]?.default;
            load();
          },
        });
        node._utilitySeedGeneratorExecuted = (seed) => {
          lastSeed = seed;
          render();
        };
        load();
        widget.options.minNodeSize = [320, 170];
        return { widget, minWidth: 320, minHeight: 170 };
      },
    };
  },
});
