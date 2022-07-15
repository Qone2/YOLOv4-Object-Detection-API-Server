window.onload = () => {
  $('#sendbutton').click(() => {
    imagebox = $('#imagebox')
    input = $('#imageinput')[0]
    if (input.files && input.files[0]) {
      let formData = new FormData();
      formData.append('images', input.files[0]);
      $.ajax({
        url: "/image/by-image-file", // fix this to your liking
        type: "POST",
        data: formData,
        cache: false,
        processData: false,
        contentType: false,
        error: function (data) {
          console.log("upload error", data);
          console.log(data.getAllResponseHeaders());
        },
        success: function (data) {
          // console.log(data);
          // bytestring = data['status']
          // image = bytestring.split('\'')[1]

          // var src = $('#imagebox').attr('src'); // "static/images/banner/blue.jpg"
          // console.log(src)
          var filename = $('input[type=file]').val().split('\\').pop();
          console.log(filename)
          var tarr = filename.split('/');      // ["static","images","banner","blue.jpg"]
          var file = tarr[tarr.length - 1]; // "blue.jpg"
          file = file.replace("jpg", "png")
          imagebox.attr('src', '..//static//detections//' + file);
          $("#link").css("display", "block");
          $("#download").attr("href", '..//static//detections//' + file);
        }
      });
    }
  });
};


function readUrl(input) {
  imagebox = $('#imagebox')
  console.log("evoked readUrl")
  if (input.files && input.files[0]) {
    let reader = new FileReader();
    reader.onload = function (e) {
      // console.log(e)

      imagebox.attr('src', e.target.result);
      imagebox.height(500);
      imagebox.width(800);
    }
    reader.readAsDataURL(input.files[0]);
  }


}
