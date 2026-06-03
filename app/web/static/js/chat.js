(function () {
  function applySwap(target, html, swap) {
    if (!target) {
      return;
    }
    if (swap === "outerHTML") {
      target.outerHTML = html;
      return;
    }
    target.innerHTML = html;
  }

  function findTarget(selector) {
    if (!selector) {
      return null;
    }
    return document.querySelector(selector);
  }

  function scrollMessages() {
    var stream = document.querySelector("[data-message-stream]");
    if (stream) {
      stream.scrollTop = stream.scrollHeight;
    }
  }

  function closeCitationPanel() {
    document.body.dataset.citationOpen = "false";
  }

  function openCitationPanel() {
    document.body.dataset.citationOpen = "true";
  }

  function scrollViewportToTop() {
    var detailCard = document.querySelector(".detail-card");
    if (detailCard) {
      detailCard.scrollTop = 0;
    }
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function handleSwap(target) {
    if (!target) {
      return;
    }
    if (target.id === "app-shell") {
      scrollMessages();
      closeCitationPanel();
    }
    if (target.id === "citation-detail") {
      openCitationPanel();
      scrollViewportToTop();
    }
  }

  function performRequest(url, options, targetSelector, swap) {
    var target = findTarget(targetSelector);
    if (!target) {
      window.location.assign(url);
      return;
    }

    fetch(url, {
      method: options.method || "GET",
      body: options.body || null,
      credentials: "same-origin",
      headers: Object.assign({ "HX-Request": "true" }, options.headers || {})
    })
      .then(function (response) {
        return response.text().then(function (html) {
          return { response: response, html: html };
        });
      })
      .then(function (result) {
        applySwap(target, result.html, swap || "innerHTML");
        handleSwap(findTarget(targetSelector));
      })
      .catch(function () {
        window.location.assign(url);
      })
      .finally(function () {
        if (options.pendingElement) {
          setPending(options.pendingElement, false);
        }
      });
  }

  function setPending(element, isPending) {
    if (!element) {
      return;
    }

    element.classList.toggle("is-requesting", isPending);
    element.classList.toggle("is-loading", isPending);
    element.setAttribute("aria-busy", isPending ? "true" : "false");

    var controls = element.querySelectorAll("button, input, textarea, select");
    controls.forEach(function (control) {
      control.disabled = isPending;
    });

    var submit = element.querySelector("button[type='submit']");
    if (!submit) {
      return;
    }
    if (isPending) {
      if (!submit.dataset.originalText) {
        submit.dataset.originalText = submit.textContent;
      }
      submit.textContent = submit.closest("#ask-form") ? "正在生成..." : "处理中...";
    } else if (submit.dataset.originalText) {
      submit.textContent = submit.dataset.originalText;
    }
  }

  document.addEventListener("DOMContentLoaded", scrollMessages);

  document.body.addEventListener("click", function (event) {
    var remote = event.target.closest("[hx-post], [hx-get]");
    if (event.target.closest("[data-close-citation]")) {
      closeCitationPanel();
      return;
    }

    if (!remote) {
      return;
    }

    if (remote.closest("form") && remote.type === "submit") {
      return;
    }
    if (remote.tagName === "FORM" && event.target !== remote) {
      return;
    }

    event.preventDefault();
    var hxPost = remote.getAttribute("hx-post");
    var hxGet = remote.getAttribute("hx-get");
    var url = hxPost || hxGet;
    if (!url) {
      return;
    }

    performRequest(
      url,
      { method: hxPost ? "POST" : "GET" },
      remote.getAttribute("hx-target"),
      remote.getAttribute("hx-swap")
    );
  });

  document.body.addEventListener("submit", function (event) {
    var form = event.target.closest("form[hx-post]");
    if (!form) {
      return;
    }

    event.preventDefault();
    var formData = new FormData(form);
    setPending(form, true);
    performRequest(
      form.getAttribute("hx-post"),
      {
        method: "POST",
        body: formData,
        pendingElement: form
      },
      form.getAttribute("hx-target"),
      form.getAttribute("hx-swap")
    );
  });
})();
