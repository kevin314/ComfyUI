import { app } from '../../../scripts/app.js'
import { setWidgetConfig } from '../../../extensions/core/widgetInputs.js'

// Helper function for chaining callbacks
function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function drawAutoAnnotated(ctx, node, widget_width, y, H) {
    console.log("Drawing AUTO widget", this.name, "isAuto:", this.isAuto);

    const litegraph_base = LiteGraph;
    const show_text = app.canvas.ds.scale >= 0.5;
    const margin = 15;

    ctx.textAlign = 'left';
    ctx.strokeStyle = litegraph_base.WIDGET_OUTLINE_COLOR;
    ctx.fillStyle = litegraph_base.WIDGET_BGCOLOR;

    ctx.beginPath();
    if (show_text && ctx.roundRect) {
        ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
    } else {
        ctx.rect(margin, y, widget_width - margin * 2, H);
    }
    ctx.fill();

    if (show_text) {
        if (!this.disabled) ctx.stroke();

        // Check if in AUTO mode
        const isAuto = this.isAuto === true;

        // Draw AUTO indicator
        ctx.save();
        ctx.font = ctx.font.replace(/\d+px/, "14px");
        if (isAuto) {
            ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
        } else {
            ctx.fillStyle = litegraph_base.WIDGET_SECONDARY_TEXT_COLOR;
        }
        ctx.fillText('AUTO', widget_width - margin - 40, y + H * 0.7);
        ctx.restore();

        // Draw label
        ctx.fillStyle = litegraph_base.WIDGET_TEXT_COLOR;
        ctx.fillText(this.label || this.name, margin * 2 + 5, y + H * 0.7);

        // Draw value
        ctx.textAlign = 'right';
        const text = isAuto ? "auto" : this.displayValue();
        ctx.fillStyle = isAuto ? litegraph_base.WIDGET_SECONDARY_TEXT_COLOR : litegraph_base.WIDGET_TEXT_COLOR;
        ctx.fillText(text, widget_width - margin - 45, y + H * 0.7);
    }

    console.log("Finished drawing AUTO widget", this.name);
}

function mouseAutoAnnotated(event, [x, y], node) {
    console.log("Mouse event on AUTO widget", this.name, event.type);

    const widget_width = node.size[0];
    const margin = 15;

    // Toggle AUTO mode when clicking on the AUTO text
    if (event.type === "pointerdown") {
        if (x > widget_width - margin - 40 && x < widget_width - margin) {
            this.isAuto = !this.isAuto;

            if (this.isAuto) {
                // Store current value before switching to auto
                this.manualValue = this.value;
            } else {
                // Restore manual value when switching from auto
                this.value = this.manualValue !== undefined ? this.manualValue : (this.options.default || 0);
            }

            // Trigger callback
            if (this.callback) {
                this.callback(this.value);
            }

            // Redraw canvas
            node.graph.setDirtyCanvas(true, false);
            return true;
        }
    }

    // Only handle other mouse events if not in AUTO mode
    if (!this.isAuto) {
        // Use default number widget behavior
        if (this.type === "FVAUTOINT") {
            return LiteGraph.widgets.number.mouse.call(this, event, [x, y], node);
        } else {
            return LiteGraph.widgets.number.mouse.call(this, event, [x, y], node);
        }
    }

    return false;
}

function makeAutoAnnotated(widget, inputData) {
    console.log("Making AUTO widget for", widget.name, "with inputData:", inputData);

    // Store original callback
    const callback_orig = widget.callback;

    // Add AUTO properties to the widget
    Object.assign(widget, {
        type: "BOOLEAN", // This is important - matches VHS.core.js approach
        draw: drawAutoAnnotated,
        mouse: mouseAutoAnnotated,
        isAuto: true,
        manualValue: widget.value,
        computeSize(width) {
            return [width, 20];
        },
        displayValue: function () {
            if (this.type === "FVAUTOINT") {
                return Math.round(this.value).toString();
            }
            return this.value.toFixed(this.options.precision || 2);
        },
        serializeValue: function () {
            return {
                value: this.value,
                isAuto: this.isAuto,
                manualValue: this.manualValue
            };
        },
        deserializeValue: function (data) {
            if (typeof data === "object" && data !== null && "isAuto" in data) {
                this.isAuto = data.isAuto;
                this.value = data.value;
                this.manualValue = data.manualValue;
            } else {
                this.value = data;
                this.isAuto = false;
            }
        },
        config: inputData,
        options: Object.assign({}, inputData[1], widget.options)
    });

    // Override callback to handle AUTO mode
    widget.callback = function (v) {
        if (this.isAuto) {
            return;
        }
        return callback_orig?.call(this, v);
    };

    // Debug: Verify the draw method is properly set
    console.log("Created AUTO widget:", widget);
    console.log("Widget draw method:", widget.draw === drawAutoAnnotated ? "Correctly set" : "NOT SET CORRECTLY");

    return widget;
}

// Register the extension with custom widgets
app.registerExtension({
    name: "FastVideo.AutoWidgets",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("beforeRegisterNodeDef called for", nodeData?.name);

        if (nodeData?.name === "VideoGenerator" || nodeData?.name === "VAEConfig" || nodeData?.name === "VAE Config") {
            console.log("Found VideoGenerator node, adding serialization support");

            // Add serialization support
            chainCallback(nodeType.prototype, "onSerialize", function (info) {
                console.log("VideoGenerator onSerialize called");
                if (!this.widgets) {
                    return;
                }

                // Ensure widgets_values exists
                if (!info.widgets_values) {
                    info.widgets_values = {};
                }

                // Handle AUTO widgets specially
                for (const w of this.widgets) {
                    if (w.type === "BOOLEAN" && w.isAuto !== undefined) {
                        info.widgets_values[w.name] = w.serializeValue();
                    }
                }
            });

            // Add deserialization support
            chainCallback(nodeType.prototype, "onConfigure", function (info) {
                console.log("VideoGenerator onConfigure called");
                if (!this.widgets || !info.widgets_values) {
                    return;
                }

                // Handle AUTO widgets specially
                for (const w of this.widgets) {
                    if (w.type === "BOOLEAN" && w.isAuto !== undefined && w.name in info.widgets_values) {
                        w.deserializeValue(info.widgets_values[w.name]);
                        w.callback?.(w.value);
                    }
                }

                // Force a redraw
                this.graph?.setDirtyCanvas(true, true);
            });

            // Override onDrawForeground to ensure our widgets are drawn
            chainCallback(nodeType.prototype, "onDrawForeground", function (ctx) {
                console.log("VideoGenerator onDrawForeground called");

                // Check if we have any AUTO widgets
                if (this.widgets) {
                    const autoWidgets = this.widgets.filter(w => w.type === "BOOLEAN" && w.isAuto !== undefined);
                    console.log("Found AUTO widgets:", autoWidgets.length);
                    console.log("widgets:", this.widgets);

                    // Debug: Check if draw methods are properly set
                    for (const w of autoWidgets) {
                        console.log(`Widget ${w.name} draw method:`, w.draw === drawAutoAnnotated ? "Correctly set" : "NOT SET CORRECTLY");
                    }
                }
            });

            // Override addInput to handle AUTO widgets
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                console.log("VideoGenerator node created, widgets:", this.widgets);

                // Convert any existing widgets to AUTO widgets if needed
                let new_widgets = [];
                if (this.widgets) {
                    for (let w of this.widgets) {
                        if (w.name === "scale_factor") {
                            new_widgets.push(makeAutoAnnotated(w, ["FVAUTOINT", { default: 8, autoDefault: 'auto' }]));
                        } else {
                            new_widgets.push(w);
                        }
                    }
                    this.widgets = new_widgets;

                    // Debug: Check if widgets were properly converted
                    const autoWidgets = this.widgets.filter(w => w.type === "BOOLEAN" && w.isAuto !== undefined);
                    console.log("After conversion, found AUTO widgets:", autoWidgets.length);
                }

                const originalAddInput = this.addInput;
                this.addInput = function (name, type, options) {
                    if (options.widget) {
                        // Is Converted Widget
                        const widget = this.widgets.find((w) => w.name == name);
                        if (widget?.config) {
                            // Has override for type
                            type = widget.config[0];
                            if (type == 'FVAUTO') {
                                type = "FLOAT,INT";
                            } else if (type == 'FVAUTOINT') {
                                type = "INT";
                            }
                            setWidgetConfig(options, widget.config);
                        }
                    }
                    return originalAddInput.apply(this, [name, type, options]);
                };

                // Force a redraw
                this.graph?.setDirtyCanvas(true, true);
            });
        }
    },

    async getCustomWidgets() {
        console.log("getCustomWidgets called for FastVideo.AutoWidgets");
        return {
            FVAUTO(node, inputName, inputData) {
                console.log("Creating FVAUTO widget for", inputName, inputData);

                // Create a standard FLOAT widget
                const widget = {
                    name: inputName,
                    type: "number",
                    value: inputData[1]?.default || 0,
                    options: inputData[1] || {},
                    callback: function (v) {
                        node.properties[inputName] = v;
                        node.graph?.setDirtyCanvas(true, false);
                    }
                };

                // Convert to AUTO widget
                return makeAutoAnnotated(widget, inputData);
            },
            FVAUTOINT(node, inputName, inputData) {
                console.log("Creating FVAUTOINT widget for", inputName, inputData);

                // Create a standard INT widget
                const widget = {
                    name: inputName,
                    type: "number",
                    value: inputData[1]?.default || 0,
                    options: Object.assign({}, inputData[1] || {}, { precision: 0 }),
                    callback: function (v) {
                        node.properties[inputName] = Math.round(v);
                        node.graph?.setDirtyCanvas(true, false);
                    }
                };

                // Convert to AUTO widget
                return makeAutoAnnotated(widget, inputData);
            }
        };
    },

    async setup() {
        console.log("FastVideo.AutoWidgets setup called");
    },

    async init() {
        console.log("FastVideo.AutoWidgets init called");

        // Force a redraw of all nodes when the extension initializes
        if (app.graph) {
            setTimeout(() => {
                app.graph.setDirtyCanvas(true, true);
            }, 1000);
        }
    }
});

console.log("FastVideo.core.js loaded");
