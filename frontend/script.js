function predictRating() {

    const review = document.getElementById("review").value;

    const result = document.getElementById("result");

    if (review.trim() === "") {

        result.innerText = "Please enter a review.";

        return;
    }

    result.innerText = "Prediction will appear here.";
}