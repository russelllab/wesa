function changeLayout(){
    var layoutValue = $("#layout-dropdown").val();

    var visibleElements = cy.elements().filter(function(element) {
        return element.visible();
    });

    layout = cy.layout({
        name: layoutValue,
        fit: true,
        eles: visibleElements
    });
    layout.run()
}

function updateGraphBasedOnThreshold(threshold) {
    cy.nodes().style("display", "element");

     if (threshold == "None") {
         if ($("#toggle_red_edges").prop("checked")) {
             var edges = cy.edges('[color!=\'#F60E0E\']');
         } else {
             var edges = cy.edges();
         }
          edges.style("display", "element");
     } else {
         // Define the thresholds
        var thresholds = [1, 5, 10, 20];
        // Iterate over each threshold
        thresholds.forEach(function (t) {
            var edges = cy.edges('[fpr' + t + '="discard"]');
            if (threshold <= t) {
                edges.style("display", "none");
            } else {
                if ($("#toggle_red_edges").prop("checked")) {
                    console.log("here")
                    var edges = cy.edges('[fpr' + t + '="discard"][color!=\'#F60E0E\']');
                }
                edges.style("display", "element");
            }
        });
     }
     hideUnconnectedNodes();
     var checked = document.getElementById("hide-unconnected-nodes").checked;
     if (checked) {
         changeLayout();
     }
}

function hideRedEdges(){
    var edges = cy.$("edge[color='#F60E0E']");
    if ( $("#toggle_red_edges").prop("checked") ) {
        edges.style("display", "none");
    } else {
        edges.style("display", "element");
    }
}

function hideUnconnectedNodes() {
    var checked = document.getElementById("hide-unconnected-nodes").checked;
    var nodes = cy.nodes().filter(function (node) {
        // Get connected edges of the node
        var connectedEdges = node.connectedEdges();

        // Check if every connected edge is hidden
        var allEdgesHidden = connectedEdges.every(function (edge) {
            return edge.style('display') === 'none';
        });

        return allEdgesHidden;
    });
    if (checked) {
        nodes.style("display", "none");
    } else {
        nodes.style("display", "element");
    }
}

function initializeGraphFeats() {

    // Modal (Help pop-up)
    var modal = document.getElementById("help-modal");
    // When the user clicks the button, open the modal
    $("#help-btn").click(function () {
        modal.style.display = "block";
    });
    // When the user clicks on (x), close the modal
    $("#help-modal-close").click(function () {
        modal.style.display = "none";
    });
    // When the user clicks anywhere outside the modal, close it
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    }

    // BUTTON: Center the graph on the page
    $("#center").click(function () {
        cy.fit(cy.$("node:visible"));
    });

    // BUTTON: Lock/Unlock element positions
    $("#lock").click(function () {
        if (cy.autolock() == false) {
            cy.autolock(true);
            $("#lock i").toggleClass("fa-unlock fa-lock");
        } else if (cy.autolock() == true) {
            cy.autolock(false);
            $("#lock i").toggleClass("fa-lock fa-unlock");
        }
    });

    // BUTTON: Zoom-In / Zoom-Out
    $("#zoom_in").click(function () {
        var z = cy.zoom() + 0.2;
        cy.zoom(z);
    });
    $("#zoom_out").click(function () {
        var z = cy.zoom() - 0.2;
        cy.zoom(z);
    });

    // BUTTON: Center the graph on the page
    $("#center").click(function () {
        cy.fit(cy.$("node:visible"));
    });

    /// DOWNLOAD BUTTONS
    $("#dl-btn").click(function () {
        if (this.className == "my-btn dl-btn-off") {
            this.className = "my-btn dl-btn-on";
            $("#dl-content").css("display", "inline-block");
        } else {
            this.className = "my-btn dl-btn-off";
            $("#dl-content").css("display", "none");
        }
    });
    // BUTTON: Download graph as JSON
    /* $("#dl_json").click(function(){
        var jsonBlob = new Blob([ JSON.stringify( cy.json() ) ], { type: "application/javascript;charset=utf-8" });
        saveAs( jsonBlob, "graph.json" );
    }); */
    // BUTTON: Get snapshot as PNG
    $("#dl_png").click(function () {
        var image = cy.png()
        var iframe = "<iframe src='" + image + `' frameborder='0'
			style='border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;'
			allowfullscreen></iframe>`;
        var win = window.open();
        win.document.write(iframe);
    });
    // BUTTON: Get snapshot as JPG
    $("#dl_jpg").click(function () {
        var image = cy.jpg()
        var iframe = "<iframe src='" + image + `' frameborder='0'
			style='border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;'
			allowfullscreen></iframe>`;
        var win = window.open();
        win.document.write(iframe);
    });
    // BUTTON: Get snapshot as SVG
    $("#dl_svg").click(function (filename) {
        var svgContent = cy.svg({scale: 1, full: true});
        var blob = new Blob([svgContent], {type: "image/svg+xml;charset=utf-8"});
        saveAs(blob, "graph.svg");
    });

    // Layout dropdown option
    $("#layout-dropdown").change(function () {
        changeLayout();
    });

    $('#fpr-threshold-slider').on('input', function () {
        var sliderValue = parseInt($(this).val(), 10);
        var fprThresholds = {1: "None", 2: 20, 3: 10, 4: 5, 5: 1}; // Mapping slider values to FPR thresholds
        var threshold = fprThresholds[sliderValue];
        if (threshold == "None") {
            $('#fpr-threshold-value').text(threshold);
        } else {
            $('#fpr-threshold-value').text('<=' + threshold + '%');
        }
        updateGraphBasedOnThreshold(threshold);
    });

    /// Toggle buttons
    $('#hide-unconnected-nodes').change(function () {
        hideUnconnectedNodes();
        changeLayout();
    });

    $("#toggle_red_edges").change(function () {
        var sliderValue = parseInt($('#fpr-threshold-slider').val(), 10);
        var fprThresholds = {1: "None", 2: 20, 3: 10, 4: 5, 5: 1}; // Mapping slider values to FPR thresholds
        var threshold = fprThresholds[sliderValue];

        updateGraphBasedOnThreshold(threshold);
        var edges = cy.$("edge[color='#F60E0E']");
        if ( $("#toggle_red_edges").prop("checked") ) {
            edges.style("display", "none");
        }
        hideUnconnectedNodes();
        changeLayout();
    });

    $("#filter_nodes").change(function () {
        var nodes = cy.$("node[node_color='#F60E0E']");
        var checked = document.getElementById("filter_nodes").checked;
        if (checked) {
            nodes.style("display", "none");
            layout.run()
        } else {
            nodes.style("display", "element");
            layout.run()
        }
    });

    $("#filter_nodes_solid").change(function () {
        var nodes = cy.$("node[nodeLinestyles='only_dashed']");
        var checked = document.getElementById("filter_nodes_solid").checked;
        if (checked) {
            nodes.style("display", "none");
        } else {
            nodes.style("display", "element");
        }
    });

    $("#hide_dashed_edges").change(function () {
        var edges = cy.$("edge[linestyle='dashed']");
        var checked = document.getElementById("hide_dashed_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
    });
}
