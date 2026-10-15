export interface Filament {
  name: string;
  hex: string;
}

export interface PaintRequest {
  provider: string;
  description: string;
  filaments: Filament[];
  reference_image: string | null; // data URL
  views: { name: string; image: string; labelled: number[] }[];
  patch_count: number;
  /** 'refine' sends renders of the current paint job for correction; 'detail' is a close-up of flagged areas. */
  mode: 'initial' | 'refine' | 'detail';
  current?: { patch: number; color: string }[];
}

export interface PaintResponse {
  plan: string;
  default_color: string;
  assignments: { patch: number; color: string }[];
  /** Regions the model wants a closer look at (eyes, teeth...). */
  detail_regions: number[];
  model: string;
  usage?: Record<string, number>;
}

export interface ProviderInfo {
  id: string;
  label: string;
  model: string;
  configured: boolean;
}

export async function listProviders(): Promise<ProviderInfo[]> {
  const r = await fetch('/api/providers');
  if (!r.ok) throw new Error(`Backend not reachable (${r.status})`);
  return r.json();
}

export async function requestPaint(req: PaintRequest, signal?: AbortSignal): Promise<PaintResponse> {
  const r = await fetch('/api/paint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok) {
    let msg = `${r.status}`;
    try { msg = (await r.json()).detail ?? msg; } catch { /* keep status */ }
    throw new Error(msg);
  }
  return r.json();
}
