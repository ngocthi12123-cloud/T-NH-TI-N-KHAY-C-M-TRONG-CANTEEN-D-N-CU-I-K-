const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";

const state = {
  app: null,
  imagePath: "",
  imageWidth: 0,
  imageHeight: 0,
  regions: [],
  regionMode: "template",
  zoom: 1,
  total: 0,
  regionMeta: {},
  customer: null,
  currentBill: null,
  selectedVoucherId: "",
};

function setStatus(text) {
  const el = $("status");
  if (el) el.textContent = text;
}

function setLoading(active) {
  $("loading").classList.toggle("hidden", !active);
}

function setInlineMessage(id, text, kind = "") {
  const el = $(id);
  if (!el) return;
  el.textContent = text || "";
  el.className = `inline-message${kind ? ` ${kind}` : ""}`;
}

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function fmtVnd(value) {
  return `${Number(value || 0).toLocaleString("en-US")} VND`;
}

async function api(url, body = null) {
  const options = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || response.statusText);
  }
  return data;
}

function optionHtml(items, valueKey = "path", labelKey = "name") {
  return items.map((item) => `<option value="${esc(item[valueKey])}">${esc(item[labelKey])}</option>`).join("");
}

async function init() {
  state.app = await api("/api/state");
  $("imageSelect").innerHTML = optionHtml(state.app.images);
  $("rewardSelect").innerHTML = (state.app.reward_catalog || []).map((reward) => (
    `<option value="${esc(reward.class_name)}">${esc(reward.display_name)} · ${reward.points_cost} điểm</option>`
  )).join("");
  renderCustomer();
  const badge = $("modelBadge");
  if (badge) {
    const classifierReady = state.app.default_model_path ? "Classifier ready" : "Classifier missing";
    badge.textContent = classifierReady;
    badge.title = "CNN dish classifier";
  }

  if (state.app.images.length) {
    await loadImage(state.app.images[0].path);
  } else {
    setStatus("No demo images found");
  }
}

function renderCustomer(message = "", kind = "") {
  const summary = $("memberSummary");
  const createButton = $("createCustomerBtn");
  if (!state.customer) {
    summary.classList.add("hidden");
    if (!message) createButton.classList.add("hidden");
    setInlineMessage("customerMessage", message || "Chưa chọn thành viên", kind);
    state.selectedVoucherId = "";
    return;
  }

  createButton.classList.add("hidden");
  summary.classList.remove("hidden");
  $("memberPhone").textContent = state.customer.phone_masked;
  $("memberPoints").textContent = `${state.customer.points_balance} điểm`;
  const activeVouchers = state.customer.active_vouchers || [];
  if (!activeVouchers.some((voucher) => voucher.id === state.selectedVoucherId)) {
    state.selectedVoucherId = "";
  }
  $("voucherSelect").innerHTML = [
    '<option value="">Không dùng voucher</option>',
    ...activeVouchers.map((voucher) => (
      `<option value="${esc(voucher.id)}">${esc(voucher.display_name)} · −${fmtVnd(voucher.discount_vnd)}</option>`
    )),
  ].join("");
  $("voucherSelect").value = state.selectedVoucherId;
  setInlineMessage("customerMessage", message || `${activeVouchers.length} voucher khả dụng`, kind || "success");
}

function renderRewardOptions() {
  const select = $("rewardSelect");
  const rewards = state.app?.reward_catalog || [];
  const balance = Number(state.customer?.points_balance || 0);
  const selectedClass = select.value;
  select.innerHTML = rewards.map((reward) => {
    const affordable = balance >= Number(reward.points_cost || 0);
    return `<option value="${esc(reward.class_name)}"${affordable ? "" : " disabled"}>${esc(reward.display_name)} · ${reward.points_cost} điểm${affordable ? "" : " · thiếu điểm"}</option>`;
  }).join("");
  const selectedReward = rewards.find((reward) => reward.class_name === selectedClass);
  if (selectedReward && balance >= Number(selectedReward.points_cost || 0)) {
    select.value = selectedClass;
    return selectedReward;
  }
  const affordableReward = rewards.find((reward) => balance >= Number(reward.points_cost || 0));
  select.value = affordableReward?.class_name || "";
  return affordableReward || null;
}

async function lookupCustomer() {
  const phone = $("phoneInput").value.trim();
  const data = await api("/api/customers/lookup", { phone });
  if (data.found) {
    state.customer = data.customer;
    $("phoneInput").value = "";
    renderCustomer("Đã tải tài khoản thành viên", "success");
  } else {
    state.customer = null;
    state.selectedVoucherId = "";
    $("createCustomerBtn").classList.remove("hidden");
    renderCustomer("Chưa có thành viên. Chọn Tạo mới để đăng ký.");
    $("createCustomerBtn").classList.remove("hidden");
  }
}

async function createCustomer() {
  const phone = $("phoneInput").value.trim();
  const data = await api("/api/customers", { phone });
  state.customer = data.customer;
  $("phoneInput").value = "";
  renderCustomer(data.created ? "Đã tạo thành viên" : "Thành viên đã tồn tại", "success");
}

function clearCustomer() {
  state.customer = null;
  state.selectedVoucherId = "";
  clearBill();
  renderCustomer("Đã chuyển sang khách vãng lai");
}

async function loadImage(path) {
  state.imagePath = path;
  setStatus("Loading image...");
  const meta = await api(`/api/image-info?path=${encodeURIComponent(path)}`);
  state.imageWidth = meta.width;
  state.imageHeight = meta.height;

  await new Promise((resolve, reject) => {
    const image = $("trayImage");
    image.onload = resolve;
    image.onerror = () => reject(new Error("Could not load tray image"));
    image.src = `/file?path=${encodeURIComponent(path)}&t=${Date.now()}`;
  });

  const titleEl = $("trayTitle");
  if (titleEl) titleEl.textContent = path.split(/[\\/]/).pop();
  state.zoom = 1;
  $("zoomInput").value = "1";
  await applyRegionsForMode();
}

function fitWidth() {
  // Image is CSS width:100% by default — get rendered width
  const img = $("trayImage");
  if (!img.naturalWidth) return Math.max(320, $("viewport").clientWidth - 30);
  return img.clientWidth || img.naturalWidth;
}

function applyZoom() {
  const image = $("trayImage");
  if (state.zoom === 1) {
    // Default: let CSS handle it (width: 100%)
    image.style.width = "";
    image.style.maxWidth = "100%";
  } else {
    // Zoomed: override CSS to fixed pixel width
    const baseWidth = $("viewport").clientWidth - 30;
    image.style.width = `${Math.round(baseWidth * state.zoom)}px`;
    image.style.maxWidth = "none";
  }
  positionOverlay();
}

function positionOverlay() {
  if (!state.imageWidth || !$("trayImage").complete) return;
  const image = $("trayImage");
  const box = $("imageBox");
  const overlay = $("overlay");
  const imageRect = image.getBoundingClientRect();
  const boxRect = box.getBoundingClientRect();
  overlay.style.left = `${imageRect.left - boxRect.left + box.scrollLeft}px`;
  overlay.style.top = `${imageRect.top - boxRect.top + box.scrollTop}px`;
  overlay.style.width = `${imageRect.width}px`;
  overlay.style.height = `${imageRect.height}px`;
  overlay.setAttribute("viewBox", `0 0 ${state.imageWidth} ${state.imageHeight}`);
}

function clampRegion(region) {
  const minSize = 24;
  region.x = Math.max(0, Math.min(state.imageWidth - minSize, Math.round(region.x)));
  region.y = Math.max(0, Math.min(state.imageHeight - minSize, Math.round(region.y)));
  region.w = Math.max(minSize, Math.min(state.imageWidth - region.x, Math.round(region.w)));
  region.h = Math.max(minSize, Math.min(state.imageHeight - region.y, Math.round(region.h)));
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

function draw() {
  applyZoom();
  const overlay = $("overlay");
  overlay.replaceChildren();

  state.regions.forEach((region, index) => {
    clampRegion(region);
    const group = svgEl("g");
    const ignored = region.label === "ignore";
    const rect = svgEl("rect", {
      class: `region-rect${ignored ? " ignored" : ""}`,
      x: region.x,
      y: region.y,
      width: region.w,
      height: region.h,
      rx: 12,
    });
    group.appendChild(rect);
    overlay.appendChild(group);
  });
}

async function applyFixedRegions() {
  if (!state.imagePath) return;
  const data = await api("/api/regions", {
    image_path: state.imagePath,
    mode: "template",
    template: "",
  });
  state.regions = data.regions;
  state.regionMeta = data;
  draw();
  setStatus(`Fixed grid ready · ${state.regions.length} regions`);
}

async function applyRegionsForMode() {
  await applyFixedRegions();
}

function animateTotal(toValue) {
  const fromValue = state.total || 0;
  const duration = 520;
  const start = performance.now();
  state.total = toValue;
  function step(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = Math.round(fromValue + (toValue - fromValue) * eased);
    $("totalValue").textContent = fmtVnd(current);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/* ── Nutrition advice engine ── */
// Map keywords found in class_name / display_name to nutrient groups
const NUTRIENT_MAP = {
  // ── Protein sources (Vietnamese food names) ──
  thit:         { protein: true },           // thịt kho, thịt kho trứng
  suon:         { protein: true },           // sườn nướng
  ca_hu:        { protein: true },           // cá hú kho
  ca:           { protein: true },           // cá (generic)
  trung:        { protein: true },           // trứng chiên
  dau_hu:       { protein: true },           // đậu hũ sốt cà
  // ── Vietnamese veggie / fibre ──
  rau:          { veggie: true },            // rau xào, canh rau
  canh:         { veggie: true },            // canh chua, canh rau
  // ── Carbs ──
  com:          { carb: true },              // cơm trắng
  bun:          { carb: true },
  mi:           { carb: true },
  // ── English fallback ──
  egg:          { protein: true },
  chicken:      { protein: true },
  pork:         { protein: true },
  fish:         { protein: true },
  rice:         { carb: true },
  vegetable:    { veggie: true },
};

function analyseNutrition(items) {
  let hasProtein = false, hasVeg = false, hasCarb = false;
  items.forEach(item => {
    if (item.ignored) return;
    const name = ((item.class_name || "") + " " + (item.display_name || "")).toLowerCase();
    for (const [keyword, flags] of Object.entries(NUTRIENT_MAP)) {
      if (name.includes(keyword)) {
        if (flags.protein) hasProtein = true;
        if (flags.veggie)  hasVeg     = true;
        if (flags.carb)    hasCarb    = true;
      }
    }
  });
  const tips = [];
  if (hasProtein && hasVeg && hasCarb) {
    tips.push({ icon: "✅", color: "ok", text: "Bữa ăn cân bằng! Đủ đạm, rau và tinh bột — tuyệt vời! 🎉" });
  } else {
    if (!hasProtein) tips.push({ icon: "🥩", color: "warn", text: "Bữa ăn thiếu đạm — hãy thêm thịt, cá, trứng hoặc đậu hũ!" });
    if (!hasVeg)     tips.push({ icon: "🥦", color: "warn", text: "Thiếu rau xanh — thêm rau củ để bổ sung chất xơ và vitamin nhé!" });
    if (!hasCarb)    tips.push({ icon: "🍚", color: "warn", text: "Chưa có tinh bột — cơm/bún/mì giúp cung cấp năng lượng cho buổi chiều!" });
    if (hasProtein && hasVeg)
      tips.push({ icon: "💪", color: "ok", text: "Đủ đạm và rau — khá tốt! Thêm chút cơm/bún để có đủ năng lượng." });
    else if (hasProtein && hasCarb)
      tips.push({ icon: "💪", color: "ok", text: "Đủ đạm và tinh bột — thêm rau để bữa ăn hoàn hảo hơn!" });
    else if (hasVeg && hasCarb)
      tips.push({ icon: "💪", color: "ok", text: "Có rau và cơm — bổ sung thêm đạm (thịt/cá/trứng) nha!" });
  }
  return tips;
}

function renderNutrition(items) {
  const el = $("nutritionAdvice");
  if (!el) return;
  const tips = analyseNutrition(items);
  el.innerHTML = tips.map(t => `
    <div class="nutrition-tip tip-${t.color}">
      <span class="tip-icon">${t.icon}</span>
      <span>${t.text}</span>
    </div>
  `).join("");
  el.style.display = "block";
}

/* ── QR Code (via qrcode.js CDN) ── */
function renderQR(bill) {
  const el = $("qrSection");
  if (!el) return;
  const qrContainer = $("qrCanvas");
  if (!qrContainer) return;
  qrContainer.innerHTML = "";

  const text = `CANTEEN BILL\nTotal: ${fmtVnd(bill.net_total_vnd ?? bill.total_vnd)}\nItems: ${bill.items.filter(i=>!i.ignored).length}\nRef: ${bill.bill_id || "checkout"}\nTime: ${new Date().toLocaleString("vi-VN")}`;

  // Use QRious or fallback to a QR API
  if (window.QRious) {
    const canvas = document.createElement("canvas");
    qrContainer.appendChild(canvas);
    new QRious({ element: canvas, value: text, size: 180, backgroundAlpha: 0, foreground: "#3a5299", level: "H" });
  } else {
    const img = document.createElement("img");
    img.src = `https://api.qrserver.com/v1/create-qr-code/?data=${encodeURIComponent(text)}&size=180x180&color=3a5299&bgcolor=ffffff&ecc=H`;
    img.alt = "Payment QR code";
    img.width = 180;
    qrContainer.appendChild(img);
  }
  el.style.display = "block";
}

function ratingSummaryText(summary) {
  const count = Number(summary?.count || 0);
  return count ? `★ ${Number(summary.average).toFixed(1)} · ${count} lượt` : "Chưa có đánh giá";
}

function ratingMarkup(item, paid) {
  const rateable = paid && !item.ignored && !item.uncertain && Number(item.price_vnd || 0) > 0;
  if (!rateable) return "";
  const selected = Number(item.rating?.stars || 0);
  const stars = [1, 2, 3, 4, 5].map((value) => (
    `<button class="star-btn${value <= selected ? " active" : ""}" type="button" data-rating-star="${value}" aria-label="${value} sao">★</button>`
  )).join("");
  return `
    <div class="rating-block" data-item-id="${esc(item.bill_item_id)}" data-stars="${selected}">
      <div class="rating-controls">
        ${stars}
        <button class="btn btn-secondary rating-save" type="button" data-save-rating>Lưu</button>
      </div>
      <textarea class="rating-comment" maxlength="500" placeholder="Nhận xét tùy chọn">${esc(item.rating?.comment || "")}</textarea>
      <div class="rating-status">${selected ? "Đã lưu đánh giá" : ""}</div>
    </div>
  `;
}

function wireRatingControls() {
  document.querySelectorAll(".rating-block").forEach((block) => {
    block.querySelectorAll("[data-rating-star]").forEach((button) => {
      button.addEventListener("click", () => {
        const stars = Number(button.dataset.ratingStar);
        block.dataset.stars = String(stars);
        block.querySelectorAll("[data-rating-star]").forEach((candidate) => {
          candidate.classList.toggle("active", Number(candidate.dataset.ratingStar) <= stars);
        });
      });
    });
    block.querySelector("[data-save-rating]").addEventListener("click", () => submitRating(block).catch(showError));
  });
}

async function submitRating(block) {
  const stars = Number(block.dataset.stars || 0);
  const status = block.querySelector(".rating-status");
  if (!stars) {
    status.textContent = "Hãy chọn số sao";
    return;
  }
  const data = await api("/api/ratings", {
    bill_item_id: block.dataset.itemId,
    customer_id: state.customer?.id || null,
    stars,
    comment: block.querySelector(".rating-comment").value,
  });
  const item = state.currentBill?.items.find((candidate) => candidate.bill_item_id === block.dataset.itemId);
  if (item) {
    item.rating = data.rating;
    item.rating_summary = data.summary;
  }
  const summary = block.closest(".bill-item").querySelector(".rating-summary");
  if (summary) summary.textContent = ratingSummaryText(data.summary);
  status.textContent = "Đã lưu đánh giá";
}

function renderRewardPanel() {
  const panel = $("rewardPanel");
  const eligible = state.currentBill?.status === "paid" && state.customer && state.currentBill.customer_id === state.customer.id;
  panel.classList.toggle("hidden", !eligible);
  if (!eligible) return;
  const alreadyIssued = Boolean(state.currentBill.voucher_issued);
  const affordableReward = renderRewardOptions();
  $("issueVoucherBtn").disabled = alreadyIssued || !affordableReward;
  setInlineMessage(
    "rewardMessage",
    alreadyIssued
      ? "Bill này đã đổi một voucher"
      : affordableReward
        ? `Số dư hiện tại: ${state.customer.points_balance} điểm`
        : `Chưa đủ điểm để đổi voucher · hiện có ${state.customer.points_balance} điểm`,
    alreadyIssued ? "success" : "",
  );
}

function renderBill(bill) {
  state.currentBill = bill;
  if (bill.customer) {
    state.customer = bill.customer;
    renderCustomer("Đã cập nhật điểm và voucher", "success");
  }
  const netTotal = Number(bill.net_total_vnd ?? bill.total_vnd ?? 0);
  animateTotal(netTotal);
  const billMetaEl = $("billMeta");
  if (billMetaEl) {
    const status = bill.status === "paid" ? "Đã thanh toán" : "Bill nháp";
    billMetaEl.textContent = `${bill.items.filter(i=>!i.ignored).length} món · ${status}`;
  }
  const jsonEl = $("billJson");
  if (jsonEl) jsonEl.textContent = JSON.stringify(bill, null, 2);
  $("billBreakdown").classList.remove("hidden");
  $("grossValue").textContent = fmtVnd(bill.gross_total_vnd ?? netTotal);
  $("discountValue").textContent = `−${fmtVnd(bill.discount_vnd || 0)}`;
  $("confirmPaymentBtn").classList.toggle("hidden", bill.status !== "draft");
  if (bill.status === "paid") {
    const message = bill.customer
      ? `Đã cộng ${bill.earned_points} điểm · Số dư ${bill.customer.points_balance} điểm`
      : "Đã xác nhận thanh toán ẩn danh";
    setInlineMessage("paymentMessage", message, "success");
  } else {
    setInlineMessage("paymentMessage", "Điểm và voucher chỉ được ghi khi xác nhận");
  }

  $("billList").classList.remove("empty-state");
  $("billList").innerHTML = bill.items.map((item, index) => {
    const confidence = Math.max(0, Math.min(1, Number(item.confidence || 0)));
    const confPct = Math.round(confidence * 100);
    const tag = item.ignored
      ? '<span class="tag muted">bỏ qua</span>'
      : item.uncertain
        ? '<span class="tag warn">chưa chắc</span>'
        : '<span class="tag ok">✓</span>';
    const finalPrice = Number(item.final_price_vnd ?? item.price_vnd ?? 0);
    const price = Number(item.discount_vnd || 0) > 0
      ? `<s>${fmtVnd(item.price_vnd)}</s>${fmtVnd(finalPrice)}`
      : fmtVnd(finalPrice);
    const summary = item.rating_summary || state.app.rating_summaries?.[item.class_name] || { average: 0, count: 0 };
    return `
      <article class="bill-item">
        <img src="${esc(item.crop_url)}" alt="${esc(item.display_name)} crop">
        <div>
          <strong>${index + 1}. ${esc(item.display_name)} ${tag}</strong>
          <div class="bill-price">${price}</div>
          <div class="confidence"><span style="width:${confPct}%"></span></div>
          <div class="subline">${esc(item.class_name)} · ${confPct}% · CNN</div>
          <div class="rating-summary">${ratingSummaryText(summary)}</div>
          ${ratingMarkup(item, bill.status === "paid")}
        </div>
      </article>
    `;
  }).join("");
  wireRatingControls();
  renderNutrition(bill.items);
  renderQR(bill);
  renderRewardPanel();
}

async function runCheckout() {
  if (!state.imagePath) return;
  setLoading(true);
  setStatus("Running checkout...");
  try {
    const thresholdInput = $("thresholdInput");
    const bill = await api("/api/run", {
      image_path: state.imagePath,
      regions: state.regions,
      region_mode: "template",
      region_metadata: state.regionMeta,
      threshold: thresholdInput ? Number(thresholdInput.value || 0.55) : 0.55,
      customer_id: state.customer?.id || null,
      voucher_id: state.selectedVoucherId || null,
    });
    renderBill(bill);
    setStatus(`✅ Đã nhận diện xong!`);
  } finally {
    setLoading(false);
  }
}

async function confirmPayment() {
  if (!state.currentBill?.bill_id || state.currentBill.status !== "draft") return;
  setLoading(true);
  setStatus("Đang xác nhận thanh toán...");
  try {
    const bill = await api("/api/checkout/confirm", { bill_id: state.currentBill.bill_id });
    renderBill(bill);
    setStatus("Đã xác nhận thanh toán");
  } finally {
    setLoading(false);
  }
}

async function issueVoucher() {
  if (!state.customer || state.currentBill?.status !== "paid") return;
  const button = $("issueVoucherBtn");
  button.disabled = true;
  setLoading(true);
  try {
    const data = await api("/api/vouchers", {
      customer_id: state.customer.id,
      source_bill_id: state.currentBill.bill_id,
      class_name: $("rewardSelect").value,
    });
    state.customer = data.customer;
    state.selectedVoucherId = data.voucher.id;
    state.currentBill.customer = data.customer;
    state.currentBill.voucher_issued = true;
    renderBill(state.currentBill);
    renderCustomer(`Đã thêm và chọn voucher ${data.voucher.display_name}`, "success");
    setStatus("Voucher đã được chọn cho bill tiếp theo");
  } finally {
    setLoading(false);
    button.disabled = Boolean(state.currentBill?.voucher_issued);
  }
}

async function applyVoucherSelection(voucherId) {
  state.selectedVoucherId = voucherId;
  if (state.currentBill?.status === "draft") {
    await runCheckout();
    setStatus(voucherId ? "Voucher đã được áp dụng vào bill" : "Đã bỏ voucher khỏi bill");
    return;
  }
  clearBill();
  setStatus(voucherId ? "Voucher sẽ áp dụng ở lần Predict tiếp theo" : "Đã bỏ chọn voucher");
}

function clearBill() {
  state.total = 0;
  state.currentBill = null;
  $("totalValue").textContent = "0 VND";
  const billMetaEl = $("billMeta");
  if (billMetaEl) billMetaEl.textContent = "Waiting for checkout";
  $("billBreakdown").classList.add("hidden");
  $("confirmPaymentBtn").classList.add("hidden");
  $("rewardPanel").classList.add("hidden");
  setInlineMessage("paymentMessage", "");
  $("billList").className = "bill-list empty-state";
  $("billList").textContent = "No items yet";
  const jsonEl = $("billJson");
  if (jsonEl) jsonEl.textContent = "{}";
  const nutr = $("nutritionAdvice");
  if (nutr) { nutr.innerHTML = ""; nutr.style.display = "none"; }
  const qrSec = $("qrSection");
  if (qrSec) qrSec.style.display = "none";
}

$("loadImageBtn").addEventListener("click", () => loadImage($("imageSelect").value).catch(showError));
$("imageSelect").addEventListener("change", () => loadImage($("imageSelect").value).catch(showError));
$("fitBtn").addEventListener("click", () => {
  state.zoom = 1;
  $("zoomInput").value = "1";
  draw();
});
$("clearBillBtn").addEventListener("click", clearBill);
$("runBtn").addEventListener("click", () => runCheckout().catch(showError));
$("lookupCustomerBtn").addEventListener("click", () => lookupCustomer().catch(showError));
$("createCustomerBtn").addEventListener("click", () => createCustomer().catch(showError));
$("clearCustomerBtn").addEventListener("click", clearCustomer);
$("confirmPaymentBtn").addEventListener("click", () => confirmPayment().catch(showError));
$("issueVoucherBtn").addEventListener("click", () => issueVoucher().catch(showError));
$("voucherSelect").addEventListener("change", (event) => applyVoucherSelection(event.target.value).catch(showError));
$("phoneInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter") lookupCustomer().catch(showError);
});
$("zoomInput").addEventListener("input", (event) => {
  state.zoom = Number(event.target.value);
  draw();
});
$("viewport").addEventListener("scroll", positionOverlay);
window.addEventListener("resize", draw);

$("uploadInput").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async () => {
    try {
      const saved = await api("/api/upload", { name: file.name, data_url: reader.result });
      state.app.images.unshift(saved);
      $("imageSelect").innerHTML = optionHtml(state.app.images);
      $("imageSelect").value = saved.path;
      await loadImage(saved.path);
    } catch (error) {
      showError(error);
    }
  };
  reader.readAsDataURL(file);
});

function showError(error) {
  setLoading(false);
  setStatus(error.message);
  alert(error.message);
}

init().catch(showError);
