$(document).ready(function () {
  $("dl.simple").each(function (i, obj) {
    $(obj).addClass("field-list");
  });

  $("ul.breatheparameterlist > li > p > .docutils").each(function (i, obj) {
    var text = obj.innerText;
    var parent = obj.parentNode;

    obj.remove();

    parent.innerHTML =
      "<strong>" + text + "</strong> - " + parent.innerHTML.slice(1);
  });
});
