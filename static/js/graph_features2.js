
$(document).ready(function(){
    /// Modal (Help pop-up)
	var modal = document.getElementById("help-modal");
	// When the user clicks the button, open the modal
	$("#help-btn").click(function() {
		modal.style.display = "block";
	});
	// When the user clicks on (x), close the modal
	$("#help-modal-close").click(function() {
		modal.style.display = "none";
	});
	// When the user clicks anywhere outside of the modal, close it
	window.onclick = function(event) {
		if (event.target == modal) {
			modal.style.display = "none";
		}
	}

	// BUTTON: Center the graph on the page
  	$("#center").click(function(){
		cy.fit( cy.$("node:visible") );
  	});

 	 // BUTTON: Lock/Unlock element positions
  	$("#lock").click(function(){
		if ( cy.autolock()==false ) {
			cy.autolock(true);
      		$("#lock i").toggleClass("fa-unlock fa-lock");
   		} else if ( cy.autolock()==true ) {
      		cy.autolock(false);
      		$("#lock i").toggleClass("fa-lock fa-unlock");
    	}
 	});

	// BUTTON: Zoom-In / Zoom-Out
	$("#zoom_in").click(function(){
		var z = cy.zoom() + 0.2;
		cy.zoom(z);
	});
	$("#zoom_out").click(function(){
		var z = cy.zoom() - 0.2;
		cy.zoom(z);
	});

	// BUTTON: Center the graph on the page
  	$("#center").click(function(){
		cy.fit( cy.$("node:visible") );
  	});

/// DOWNLOAD BUTTONS
	$("#dl-btn").click(function(){
	if ( this.className == "my-btn dl-btn-off" ) {
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
  	$("#dl_png").click(function(){
		var image = cy.png()
		var iframe = "<iframe src='"+image+`' frameborder='0'
			style='border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;'
			allowfullscreen></iframe>`;
  		var win = window.open();
	 	win.document.write(iframe);
  	});
  	// BUTTON: Get snapshot as JPG
 	$("#dl_jpg").click(function(){
		var image = cy.jpg()
		var iframe = "<iframe src='"+image+`' frameborder='0'
			style='border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;'
			allowfullscreen></iframe>`;
  		var win = window.open();
	  	win.document.write(iframe);
	 });
	// BUTTON: Get snapshot as SVG
  	$("#dl_svg").click(function(filename) {
		var svgContent = cy.svg({scale: 1, full: true});
		var blob = new Blob([svgContent], {type:"image/svg+xml;charset=utf-8"});
		saveAs(blob, "graph.svg");
	});

/// Toggle buttons
    $("#toggle_red_edges").change(function(){
        var edges = cy.$("edge[color='#F60E0E']");
        var checked = document.getElementById("toggle_red_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
  	});

  	$("#filter_nodes").change(function(){
        var nodes = cy.$("node[node_color='#F60E0E']");
        var checked = document.getElementById("filter_nodes").checked;
        if (checked) {
            nodes.style("display", "none");
            //$("#elm_red_container").show();
        } else {
            nodes.style("display", "element");
            //$("#elm_red_container").hide();
            //$("#toggle_red_edges").prop("checked", false);
        }
  	});

  	$("#filter_nodes_solid").change(function(){
        var nodes = cy.$("node[nodeLinestyles='only_dashed']");
        var checked = document.getElementById("filter_nodes_solid").checked;
        if (checked) {
            nodes.style("display", "none");
        } else {
            nodes.style("display", "element");
        }
  	});

  	$("#hide_dashed_edges").change(function(){
        var edges = cy.$("edge[linestyle='dashed']");
        var checked = document.getElementById("hide_dashed_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
  	});

  	$("#hide_fpr1_edges").change(function(){
        var edges = cy.$("edge[fpr1='discard']");
        var checked = document.getElementById("hide_fpr1_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
  	});
  	$("#hide_fpr5_edges").change(function(){
        var edges = cy.$("edge[fpr5='discard']");
        var checked = document.getElementById("hide_fpr5_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
  	});
  	$("#hide_fpr10_edges").change(function(){
        var edges = cy.$("edge[fpr10='discard']");
        var checked = document.getElementById("hide_fpr10_edges").checked;
        if (checked) {
            edges.style("display", "none");
        } else {
            edges.style("display", "element");
        }
  	});
  	$("#hide_fpr20_edges").change(function(){
        var edges = cy.$("edge[fpr20='discard']");
        var checked = document.getElementById("hide_fpr20_edges").checked;
        if (checked) {
            edges.style("display", "none");
            $("#elm_confirmed_container").show();
        } else {
            edges.style("display", "element");
            $("#elm_confirmed_container").hide();
			$("#hide_fpr10_edges").prop("checked", false);
			$("#hide_fpr5_edges").prop("checked", false);
			$("#hide_fpr1_edges").prop("checked", false);
        }
  	});

	cy.on("click", "node", function(event) {
		var node = event.target;
		var name = node.data("id");
		var comp = node.data("complexes");
		var nr_comp = node.data("number-of-complexes");
        var qtip_content = "<span class='tip'>"+name+"<br>\n"+
                        "\t<b>Number of complexes: </b>"+nr_comp+"<br>\n" +
                        "\t<b>Complex names: </b>"+comp+"<br>\n" +
                        "</span>"
		node.qtip({
			content: qtip_content,
			position: {
				my: "top center",
				at: "bottom center"
			},
			style: {
				classes: "qtip-bootstrap",
				tip: {
					width: 20,
					height: 10
				}
			},
			show: { event: "directtap" }
		});
		this.trigger("directtap");
	});

	cy.on("click", "edge", function(event) {
		var edge = event.target;
		var source = edge.data("source");
		var target = edge.data("target");
		var wesa = edge.data("WeSA");
		var sa = edge.data("SA");
		var Ox = edge.data("Observed: bait = source");
		var Oy = edge.data("Observed: bait = target");
		var Om = edge.data("Observed matrix");

        var qtip_content = "<span class='tip'>"+source + " -- " + target +"<br>\n"+
                        "\t<b>WeSA score: </b>"+wesa+"<br>\n" +
                        "\t<b>SA score: </b>"+sa+"<br>\n" +
                        "\t<b>Experiments with the first protein as bait: </b>"+Ox+"<br>\n" +
                        "\t<b>Experiments with the second protein as bait: </b>"+Oy+"<br>\n" +
                        "\t<b>Experiments where both were in the matrix: </b>"+Om+"<br>\n" +
                        "</span>"
		edge.qtip({
			content: qtip_content,
			position: {
				my: "bottom center",
				at: "bottom center",
				adjust: {
                    screen: true
                }
			},
			style: {
				classes: "qtip-bootstrap",
				tip: {
					width: 20,
					height: 10
				}
			},
			show: { event: "directtap" }
		});
		this.trigger("directtap");
	});
});