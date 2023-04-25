document.addEventListener("DOMContentLoaded", function(event) {
                let delinq = document.getElementById('delinq')
                let appr = document.getElementById('appr')
                let depr = document.getElementById('depr')
                let flag_lines = document.getElementsByClassName("minor_line");
                for(let i=0; i<2; i++){
                    let checkIcon = flag_lines[i].previousElementSibling;
                    if(i==0){
                        if(delinq.textContent.includes("True")){
                            checkIcon.src="/static/icons/x-mark.png";
                            flag_lines[i].textContent = "Warning: this mortgage has a history of delinquency.";
                        }
                        else{
                            checkIcon.src="/static/icons/check.png";
                            flag_lines[i].textContent = "This mortgage does not have a history of delinquency.";
                        }
                    }
                    else {
                        if (appr.textContent.includes("True")) {
                            checkIcon.src = "/static/icons/check.png";
                            flag_lines[i].textContent = "This property has significantly appreciated in value!";
                        }
                        else if(depr.textContent.includes("True")){
                            checkIcon.src = "/static/icons/x-mark.png";
                            flag_lines[i].textContent = "This property has significantly depreciated in value.";
                        }
                        else{
                            checkIcon.src = "/static/icons/check.png";
                            flag_lines[i].textContent = "This property has not significantly changed in value.";
                        }
                    }
                }
            });
