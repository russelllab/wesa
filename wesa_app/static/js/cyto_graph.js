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

    }).fail(function() {
        console.error("Error loading Cytoscape data.");
    });
}
