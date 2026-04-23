```html
<script setup>
import { ref } from 'vue'

const searchQuery = ref('')
const results = ref([])
const isSearching = ref(false)
const filtersEnabled = false 
// --- YENİ: Modüler Filtre Deposu ---
const activeFilters = ref({
  konu: ''
})
const staticKonular = ['Tazminat', 'Alacak', 'Fikri Haklar', 'İş Hukuku', 'Ticari Davalar']

// Modal için gerekli değişkenler
const isModalOpen = ref(false)
const selectedKarar = ref(null)

const handleSearch = async () => {
  isSearching.value = true
  try {
    const url = new URL('http://127.0.0.1:5000/api/search')
    
    if (searchQuery.value) {
      url.searchParams.append('q', searchQuery.value)
    }

    Object.entries(activeFilters.value).forEach(([key, value]) => {
      if (value) {
        url.searchParams.append(key, value)
      }
    })

    const response = await fetch(url)
    const data = await response.json()
    results.value = data
  } catch (error) {
    console.error("Bağlantı hatası:", error)
    alert("Backend'e bağlanılamadı. Flask terminalinin çalıştığından emin ol.")
  } finally {
    isSearching.value = false
  }
}

// --- YENİ: Dinamik Filtreleme Fonksiyonu ---
const toggleFilter = (filterKey, value) => {
  if (activeFilters.value[filterKey] === value) {
    activeFilters.value[filterKey] = '' // Zaten seçiliyse iptal et
  } else {
    activeFilters.value[filterKey] = value // Değilse seç
  }
  handleSearch() // Seçim değişince aramayı tetikle
}

// Detay butonuna basınca çalışacak fonksiyon
const openModal = (karar) => {
  selectedKarar.value = karar
  isModalOpen.value = true
}

const closeModal = () => {
  isModalOpen.value = false
  selectedKarar.value = null
}
</script>

<template>
  <div class="min-h-screen bg-slate-50 text-slate-900 font-sans relative">
    
    <header class="p-6 border-b bg-white flex justify-between items-center shadow-sm">
      <div class="flex items-center gap-3">
        <img src="./assets/logo.png" alt="hukuki_asistan Logo" class="h-10 w-auto object-contain" />
        <h1 class="text-2xl font-bold text-law-blue tracking-tight">Hukuki Asistan<span class="text-sm font-normal text-slate-500"></span></h1>
      </div>

      <div class="space-x-4">
        <button class="text-slate-600 hover:text-law-blue font-medium transition-colors">Geçmiş</button>
        <button class="bg-law-blue text-white px-5 py-2 rounded-lg shadow-md hover:bg-slate-800 transition-all font-medium">Giriş Yap</button>
      </div>
    </header>

    <main class="max-w-4xl mx-auto mt-20 px-4 pb-20">
      <div class="text-center mb-12">
        <h2 class="text-4xl font-extrabold mb-4 text-slate-800 tracking-tight">Hukuki Emsal Karar Arama</h2>
        <p class="text-slate-500 text-lg">Yapay zeka destekli, hibrit arama teknolojisi ile emsallere saniyeler içinde ulaşın.</p>
      </div>

      <div class="relative shadow-2xl rounded-2xl group">
        <input 
          v-model="searchQuery"
          @keyup.enter="handleSearch"
          type="text" 
          placeholder="Örn: Eser sahipliği ve manevi hak ihlali..."
          class="w-full p-6 pr-32 rounded-2xl border-2 border-transparent focus:border-law-blue/30 outline-none text-lg transition-all"
        />
        <button 
          @click="handleSearch" 
          :disabled="isSearching"
          class="absolute right-3 top-3 bottom-3 bg-law-blue text-white px-8 rounded-xl hover:bg-slate-800 transition-all font-semibold disabled:opacity-70"
        >
          {{ isSearching ? 'Aranıyor...' : 'Ara' }}
        </button>
      </div>

      <div v-if="filtersEnabled" class="flex flex-wrap justify-center gap-2 mt-4">
        <button 
          v-for="konu in staticKonular" 
          :key="konu"
          @click="toggleFilter('konu', konu)"
          :class="[
            'px-4 py-1.5 rounded-full text-sm font-medium transition-all border',
            activeFilters.konu === konu 
              ? 'bg-law-blue text-white border-law-blue shadow-md scale-105' 
              : 'bg-white text-slate-600 border-slate-200 hover:border-law-blue/30 hover:bg-slate-50'
          ]"
        >
          {{ konu }}
        </button>

        <button 
          v-if="activeFilters.konu" 
          @click="toggleFilter('konu', activeFilters.konu)"
          class="text-xs text-red-500 hover:underline ml-2"
        >
          Filtreyi Kaldır
        </button>
      </div>
      
      <div v-if="results.length > 0" class="mt-12 space-y-6">
        <h3 class="text-slate-500 font-medium mb-4">{{ results.length }} sonuç bulundu</h3>
        
        <div v-for="karar in results" :key="karar.id" class="bg-white p-8 rounded-2xl shadow-sm border border-slate-100 hover:shadow-lg transition-all text-left group">
          <div class="flex justify-between items-start mb-4">
            <div>
              <span class="bg-blue-50 text-law-blue px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase">{{ karar.esas_no }}</span>
              <h3 class="text-xl font-bold mt-3 text-slate-800 group-hover:text-law-blue transition-colors">{{ karar.mahkeme }}</h3>
            </div>
            <span class="text-slate-400 text-sm font-medium">{{ karar.id }}</span>
          </div>
          
          <p class="text-law-blue font-semibold mb-4">{{ karar.konu }}</p>
          
          <div class="bg-slate-50 p-5 rounded-xl text-sm text-slate-600 mb-6 border border-slate-100 leading-relaxed">
            <strong class="text-slate-800 block mb-2 text-base">Olay Özeti:</strong> 
            {{ karar.olay_ozeti }}
          </div>
          
          <div class="flex justify-between items-center border-t border-slate-100 pt-4">
            <span class="font-bold text-emerald-600">{{ karar.hibrit_skor || '0.00' }}</span>
            <button @click="openModal(karar)" class="text-law-blue font-bold hover:underline text-sm flex items-center gap-1">
              Hükmü ve Detayları Gör 
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clip-rule="evenodd" /></svg>
            </button>
          </div>
        </div>
      </div>
    </main>

    <div v-if="isModalOpen" class="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div @click="closeModal" class="absolute inset-0 bg-slate-900/40 backdrop-blur-sm transition-opacity"></div>
      
      <div class="relative bg-white rounded-3xl shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        
        <div class="px-8 py-6 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
          <div>
            <span class="bg-law-blue text-white px-3 py-1 rounded-full text-xs font-bold tracking-wider uppercase mb-2 inline-block">{{ selectedKarar.esas_no }}</span>
            <h3 class="text-xl font-bold text-slate-800">{{ selectedKarar.mahkeme }}</h3>
          </div>
          <button @click="closeModal" class="text-slate-400 hover:text-red-500 transition-colors p-2 rounded-full hover:bg-red-50">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
          </button>
        </div>
        
        <div class="p-8 overflow-y-auto">
          <h4 class="font-extrabold text-law-blue mb-4 text-lg">Karar Hükmü</h4>
          <div class="prose prose-slate max-w-none">
            <p class="text-slate-700 leading-relaxed whitespace-pre-wrap">{{ selectedKarar.hukum }}</p>
          </div>
          
          <div v-if="selectedKarar.xai_vurgular && selectedKarar.xai_vurgular.length > 0" class="mt-8 p-5 bg-emerald-50 rounded-xl border border-emerald-100 flex items-start gap-4">
            <div class="bg-emerald-100 p-2 rounded-full text-emerald-600 mt-1">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
            </div>
            <div class="flex-1">
              <p class="text-sm text-emerald-800 font-bold mb-2">XAI (Açıklanabilir Yapay Zeka) Analizi</p>
              <p class="text-xs text-emerald-700 mb-3 font-medium">Model, aşağıdaki karar cümlelerini arama sorgunuzla en yüksek vektörel eşleşme gösterdiği için öne çıkardı:</p>
              
              <ul class="space-y-3">
                <li v-for="(cumle, index) in selectedKarar.xai_vurgular" :key="index" class="text-sm text-slate-800 bg-white p-3 rounded-lg border border-emerald-100 shadow-sm relative pl-8">
                  <span class="absolute left-3 top-3 text-emerald-400 font-serif text-2xl leading-none">"</span>
                  {{ cumle }}
                </li>
              </ul>
            </div>
          </div>
        </div> 
      </div>
    </div>

  </div>
</template>
```