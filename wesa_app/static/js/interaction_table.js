"use strict";

function initializeTable() {
    var table = $('#interaction-table').DataTable({
        dom: "lBfrtip",
        buttons: ['copy', 'csv', 'excel', 'pdf', 'print'],
        order: [[2, 'desc']],
        "ajax": ints_json,
        "scrollX": true
    });
}
