"use strict";

$(document).ready(function(){
      //$(function() {
            //var dataset = {{ints_json}}
            //var columns = {{columns}}
            var table = $('#interaction-table').DataTable( {
                    "ajax": ints_json,
                    //columns: columns,
                    dom: "lBfrtip",
                    buttons: ['copy', 'csv', 'excel', 'pdf', 'print'],
                    order: [[2, 'desc']],

                    //columns: columns
                    "scrollX": true//,
                    /*"columnDefs": [
                            //adds links to columns with UniProt accession
                            {
                                    targets: [2, 5],
                                    "render": function ( data, type, row, meta ) {
                                            return '<a href=http://www.uniprot.org/uniprot/'+data+'  target="_blank">'+data+'</a>';
                                            }
                            },
                            {
                                    targets: [-2],
                                    "visible": false,
                                    "searchable": false
                            }
                    ],
                    "initComplete": function(settings){
                            $('#interaction-table thead th, #interaction-table tfoot th').each(function () {
                                    var td = $(this);
                                    var headerText = td.text();
                            });


                            //show only when hovering within Type column
                            $('#interaction-table tr td:nth-child(3)').each(function () {
                                    var td = $(this);
                                    var headerText = td.text();
                                    var headerTitle= "";
                                    if ( headerText == "PROT::PROT" ) headerTitle =  "Protein-Protein interaction; supported by experimental evidence from BioGRID";
                                    else if (headerText == "DOM::DOM" ) headerTitle = "Domain-Domain interaction; derived via HiRes 3D structure of interaction domains from 3did";
                                    else if (headerText == "iDOM::iDOM" ) headerTitle = "Domain-Domain interaction; predicted from protein signatures";
                                    else if (headerText == "DOM::ELM" ) headerTitle = "Domain-Linear Motif interaction; from ELM database";
                                    else if (headerText == "ELM::DOM" ) headerTitle = "Domain-Linear Motif interaction; from ELM database";
                                    else if (headerText == "InterPreTS" ) headerTitle = "Region-Region interaction; based on homology with a structure of interacting proteins, predicted with InterPreTS";
                                    td.attr('title', headerTitle);
                            });

                            $('#interaction-table thead th[title]').tooltip({
                                    "container": 'body',
                                    "delay": 0,
                                    "track": true,
                                    "fade": 250
                            });
                }*/
           });
    });

/*$(document).ready(function(){
        var table = $('#table').DataTable( {
                dom: "lBfrtip",
                buttons: ['copy', 'csv', 'excel', 'pdf', 'print'],
                "ajax": {{data|tojson}},
                "scrollX": true,
                "columnDefs": [ //{{columns|tojson}}
                        //adds links to columns with UniProt accession
                        {
                                targets: [2, 5],
                                "render": function ( data, type, row, meta ) {
                                        return '<a href=http://www.uniprot.org/uniprot/'+data+'  target="_blank">'+data+'</a>';
                                        }
                        },
                        {
                                targets: [-2],
                                "visible": false,
                                "searchable": false
                        }
                ],
                "initComplete": function(settings){
                        $('#interaction-table thead th, #interaction-table tfoot th').each(function () {
                                var td = $(this);
                                var headerText = td.text();
                        });


                        //show only when hovering within Type column
                        $('#interaction-table tr td:nth-child(3)').each(function () {
                                var td = $(this);
                                var headerText = td.text();
                                var headerTitle= "";
                                if ( headerText == "PROT::PROT" ) headerTitle =  "Protein-Protein interaction; supported by experimental evidence from BioGRID";
                                else if (headerText == "DOM::DOM" ) headerTitle = "Domain-Domain interaction; derived via HiRes 3D structure of interaction domains from 3did";
                                else if (headerText == "iDOM::iDOM" ) headerTitle = "Domain-Domain interaction; predicted from protein signatures";
                                else if (headerText == "DOM::ELM" ) headerTitle = "Domain-Linear Motif interaction; from ELM database";
                                else if (headerText == "ELM::DOM" ) headerTitle = "Domain-Linear Motif interaction; from ELM database";
                                else if (headerText == "InterPreTS" ) headerTitle = "Region-Region interaction; based on homology with a structure of interacting proteins, predicted with InterPreTS";
                                td.attr('title', headerTitle);
                        });

                        // Apply the tooltips
                        $('#interaction-table thead th[title]').tooltip({
                                "container": 'body',
                                "delay": 0,
                                "track": true,
                                "fade": 250
                        });

            }
        });
}); */