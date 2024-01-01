"use strict";

var json_elements = (function () {
  var json_elements = null;
  $.ajax({
      'async': false,
      'global': false,
      'url': graph_json,
      'dataType': "json",
      'success': function (data) {
        json_elements = data;
      }
  });
  return json_elements;
})();

// var json_style = (function () {
//   var json_style = null;
//   $.ajax({
//       'async': false,
//       'global': false,
//       'url': style_json,
//       'dataType': "json",
//       'success': function (data) {
//         json_style = data;
//       }
//   });
//   return json_style;
// })();

var json_style = [
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


var cy;
document.addEventListener("DOMContentLoaded", function() {
  cy = cytoscape({
        container: document.getElementById("cy"),
        elements: json_elements,
    style: json_style,
        layout: {
        //name: 'preset',
        name: 'cose',
        fit: true
        },
    // motionBlur: true,
    minZoom: 0.05,
    maxZoom : 3.0,
    selectionType: 'single',
    wheelSensitivity: 0.3,
    // pixelRatio: 1,
    hideEdgesOnViewport: false, // These 2 options could be used only if network is big enough
    // textureOnViewport: true
        });

  cy.fit( cy.$("node:visible") );
});