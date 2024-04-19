"use strict";

var cy;
var layout;

function initializeCytoscape() {

    let json_style = [
        {
            "selector": "node",
            "style": {
                "width": "25px",
                "height": "12px",
                "shape": "roundrectangle",
                "background-opacity": "1",
                "background-color": "data(node_color)",
                "border-width": "0.1px",
                "border-color": "black",
                "border-opacity": "1",
                "label": "data(id)",
                "color": "white",
                "font-size": "5.5px",
                "font-weight": "bold",
                "text-outline-width": "0.3",
                "text-outline-color": "data(node_color)",
                "text-halign": "center",
                "text-valign": "bottom",
                "text-margin-y": "-8.5px",
                "events": "yes",
                "text-events": "yes"
            }
        },
        {
            "selector": "edge",
            "style": {
                "line-color": "data(color)",
                "line-style": "data(linestyle)",
                "opacity": "0.7",
                "width": "data(linewidth)"
            }
        }
    ]

    // Use jQuery's getJSON to load json_elements asynchronously
    $.getJSON(graph_json, function(json_elements) {
        cy = cytoscape({
            container: document.getElementById("cy"),
            elements: json_elements,
            style: json_style, // Ensure json_style is defined and accessible
            layout: {
                name: 'cose',
                fit: true
            },
            minZoom: 0.05,
            maxZoom: 3.0,
            selectionType: 'single',
            wheelSensitivity: 0.3,
        });

        cy.fit(cy.$("node:visible"));

        layout = cy.layout({
            name: 'cose',
            fit: true
        });

        cy.nodes().qtip({
            content: function() {
                var name = this.data("id");
                var comp = this.data("complexes");
                var nr_comp = this.data("number-of-complexes");
                var qtip_content = "<b>Protein: " + name + "</b><br>\n" +
                    "<b>Number of complexes: </b>" + nr_comp + "<br>\n" +
                    "<b>Complex names: </b>" + comp + "<br>\n"
                return qtip_content
            },
            position: {
                my: "top center",
                at: "bottom center"
            },
            show: {
                event: "click"
            },
            style: {
                classes: "qtip-bootstrap",
                tip: {
                    width: 20,
                    height: 10
                }
            }
        });

        cy.edges().qtip({
            content: function() {
                var source = this.data("source");
                var target = this.data("target");
                var wesa = this.data("WeSA");
                var sa = this.data("SA");
                var Ox = this.data("Observed: bait = source");
                var Oy = this.data("Observed: bait = target");
                var Om = this.data("Observed matrix");
                var qtip_content = "<b>Protein Pair: <span style='color: #1f618d;'>" + source + "</span> -- <span style='color: #b9770e'>" + target + "</span></b><br>\n" +
                    "<span style='float:left; width: 100px; font-weight: bold'>WeSA score: </span>" + wesa + "<br>\n" +
                    "<span style='float:left; width: 100px; font-weight: bold'>SA score: </span>" + sa + "<br>\n" +
                    "<span style='float:left; width: 280px; font-weight: normal'>Experiments with the <span style='color: #1f618d;'>first protein</span> as bait: </span>" + Ox + "<br>\n" +
                    "<span style='float:left; width: 280px; font-weight: normal'>Experiments with the <span style='color: #b9770e'>second protein</span> as bait: </span>" + Oy + "<br>\n" +
                    "<span style='float:left; width: 280px; font-weight: normal'>Experiments where both were in the matrix: </span>" + Om + "<br>\n"
                return qtip_content
            },
            position: {
                my: "top center",
                at: "bottom center"
            },
            show: {
                event: "click"
            },
            style: {
                classes: "qtip-bootstrap",
                tip: {
                    width: 20,
                    height: 10
                }
            }
        });

    }).fail(function() {
        console.error("Error loading Cytoscape data.");
    });
}
