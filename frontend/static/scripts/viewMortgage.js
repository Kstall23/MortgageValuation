document.addEventListener("DOMContentLoaded", function(event) {
                let questionIcon = document.getElementById("question");
                let additionalInfo = document.getElementById("explore");
                let delinq = document.getElementById('delinq')
                let appr = document.getElementById('appr')
                let depr = document.getElementById('depr')
                let flag_values = [delinq, appr, depr]
                questionIcon.onclick = function(){
                    additionalInfo.style.display="block";
                }
                let flag_lines = document.getElementsByClassName("minor_line");
                for(let i=0; i<3; i++){
                    let checkIcon = flag_lines[i].previousElementSibling;
                    if(i==0){
                        if(flag_values[i].textContent.includes("True")){
                            checkIcon.src="/static/icons/x-mark.png";
                            flag_lines[i].textContent = "Warning: this mortgage has a history of delinquency.";
                        }
                        else{
                            checkIcon.src="/static/icons/check.png";
                            flag_lines[i].textContent = "This mortgage does not have a history of delinquency.";
                        }
                    }
                    else if(i==1) {
                        if (flag_values[i].textContent.includes("True")) {
                            checkIcon.src = "/static/icons/check.png";
                            flag_lines[i].textContent = "This mortgage has significantly appreciated in value!";
                        } else {
                            checkIcon.src = "/static/icons/x-mark.png";
                            flag_lines[i].textContent = "This mortgage has not significantly appreciated in value.";
                        }
                    }
                    else {
                        if (flag_values[i].textContent.includes("True")) {
                            checkIcon.src = "/static/icons/x-mark.png";
                            flag_lines[i].textContent = "Warning: this mortgage has significantly depreciated in value!";
                        } else {
                            checkIcon.src = "/static/icons/check.png";
                            flag_lines[i].textContent = "This mortgage has not significantly depreciated in value.";
                        }
                    }
                }
            });
