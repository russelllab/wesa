"use strict";

$(document).ready(function(){

    // validate input form on keyup and submit
    $("#form").validate({
            /*rules: {
                    prots_input: "required",
            },*/
            messages: {
                    prots_input: "Please enter your proteins in the input box.\nIf you selected a local file, you need to upload it first, then submit.",
                    //max_prots: "The maximum number of interactors will be defaulted to 5.",
            }
    });

    $("#all_intrs").click(function() {
            var input = document.getElementById('add_n_interactors');
            if (this.checked) {
                    input.disabled = true;
            } else {
                    input.disabled= false;
            }
    });

// Examples
    $("#ex1").click(function() {
        $("#prots_input").val(`POLR2K
            POLR2A
            POLR2B
            POLR2C
            POLR3A
            POLR3C
            POLR1D
            POLR1A`);
        $("#DataSource").val("intact");
    });

    $("#ex2").click(function() {
            $("#prots_input").val(`BBS7
                BBS10
                LZTFL1`);
            });

    $("#ex3").click(function() {
                $("#prots_input").val(`IFT172 IFT80
                    IFT172 IFT57
                    IFT172 LCA5
                    IFT172 KATNIP
                    IFT172 TRAF3IP1
                    IFT172 IFT88
                    IFT172 IFT52
                    IFT172 IFT20`);
            });

    $("#ex4").click(function() {
            $("#prots_input").val(`CCT3 CCT2 experiment01
                CCT3 CCT4 experiment01
                CCT4 CCT2 experiment02
                IFT57 CLUAP1 experiment03
                IFT88 IFT70 experiment04
                IFT88 IFT57 experiment04
                IFT88 IFT74 experiment04
                IFT57 IFT88 experiment05
                IFT88 IFT57 experiment06`);
            });

    $("#ex5").click(function() {
            $("#prots_input").val(`TTC30B
                CLUAP1
                IFT172
                IFT80
                IFT20
                TRAF3IP1
                IFT57
                IFT52
                TTC30A
                IFT88
                IFT46
                IFT27
                IFT22
                HSPB11
                IFT81
                IFT74
                TTC26`);
            });
});